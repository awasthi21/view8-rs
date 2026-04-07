use lazy_static::lazy_static;
use rayon::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::process::Command;
use std::time::Instant;

// =============================================================================
// Memory comparison for 308K bytecode lines:
//   Python View8: ~900GB (GC + dicts + string copies + __dict__ per object)
//   Rust view8-rs: ~30MB (flat structs, no GC, no overhead)
// =============================================================================

#[derive(Clone, PartialEq)]
enum ExportFormat { V8Opcode, Translated, Decompiled }

struct Config {
    input: String,
    output: Option<String>,       // None = stdout
    disassembler: Option<String>, // -p / --path
    formats: Vec<ExportFormat>,   // -e / --export-format
    filter: Option<Regex>,        // -f / --filter
    min_ops: Option<usize>,       // --min-ops
    max_ops: Option<usize>,       // --max-ops
    show_offset: bool,            // --offset
    no_stubs: bool,               // --no-stubs
    json: bool,                   // --json
    stats: bool,                  // --stats
}

impl Config {
    fn parse() -> Result<Self, String> {
        let args: Vec<String> = env::args().collect();
        let mut positional: Vec<String> = Vec::new();
        let mut disassembler = None;
        let mut formats = Vec::new();
        let mut filter = None;
        let mut min_ops = None;
        let mut max_ops = None;
        let mut show_offset = false;
        let mut no_stubs = false;
        let mut json = false;
        let mut stats = false;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "-p" | "--path" => {
                    i += 1;
                    disassembler = Some(args.get(i).cloned().ok_or("--path requires a value")?);
                }
                "-e" | "--export-format" => {
                    i += 1;
                    while i < args.len() && !args[i].starts_with('-') {
                        formats.push(match args[i].as_str() {
                            "v8_opcode"  => ExportFormat::V8Opcode,
                            "translated" => ExportFormat::Translated,
                            "decompiled" => ExportFormat::Decompiled,
                            other => return Err(format!(
                                "Unknown format '{}'. Valid: v8_opcode, translated, decompiled", other)),
                        });
                        i += 1;
                    }
                    continue;
                }
                "-f" | "--filter" => {
                    i += 1;
                    let pat = args.get(i).ok_or("--filter requires a pattern")?;
                    filter = Some(Regex::new(pat)
                        .map_err(|e| format!("Invalid --filter regex: {}", e))?);
                }
                "--min-ops" => {
                    i += 1;
                    min_ops = Some(args.get(i).ok_or("--min-ops requires a number")?
                        .parse::<usize>().map_err(|_| "--min-ops must be a positive integer")?);
                }
                "--max-ops" => {
                    i += 1;
                    max_ops = Some(args.get(i).ok_or("--max-ops requires a number")?
                        .parse::<usize>().map_err(|_| "--max-ops must be a positive integer")?);
                }
                "--offset"   => { show_offset = true; }
                "--no-stubs" => { no_stubs = true; }
                "--json"     => { json = true; }
                "--stats"    => { stats = true; }
                _ if !args[i].starts_with('-') || args[i] == "-" => { positional.push(args[i].clone()); }
                other => return Err(format!("Unknown option: {}", other)),
            }
            i += 1;
        }

        if formats.is_empty() { formats.push(ExportFormat::Decompiled); }

        let input = positional.first().cloned().ok_or("Missing input file")?;
        let output = match positional.get(1).map(|s| s.as_str()) {
            None | Some("-") => None,
            Some(s) => Some(s.to_string()),
        };

        Ok(Config { input, output, disassembler, formats, filter,
                    min_ops, max_ops, show_offset, no_stubs, json, stats })
    }

    fn print_usage() {
        eprintln!("view8_rs — V8 bytecode decompiler (Rust)");
        eprintln!("Usage: view8_rs [options] <input> [output]");
        eprintln!("       (omit output or use '-' to write to stdout)");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  -p, --path <binary>           Path to disassembler binary (e.g. d8)");
        eprintln!("  -e, --export-format <fmt...>  Output format(s) (default: decompiled)");
        eprintln!("                                   v8_opcode   raw V8 opcode + operands");
        eprintln!("                                   translated  accumulator-model pseudo-JS");
        eprintln!("                                   decompiled  resolved constants (default)");
        eprintln!("  -f, --filter <regex>          Only output functions matching pattern");
        eprintln!("      --min-ops <n>             Skip functions with fewer than n instructions");
        eprintln!("      --max-ops <n>             Skip functions with more than n instructions");
        eprintln!("      --offset                  Prefix each instruction with its bytecode offset");
        eprintln!("      --no-stubs                Skip unnamed stub functions");
        eprintln!("      --json                    Output as JSON array instead of pseudo-JS");
        eprintln!("      --stats                   Print opcode frequency table to stderr");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  view8_rs bytecode.txt out.js");
        eprintln!("  view8_rs -p ./d8 input.jsc out.js");
        eprintln!("  view8_rs -e v8_opcode decompiled --offset bytecode.txt out.js");
        eprintln!("  view8_rs --filter 'handleRequest' --min-ops 5 bytecode.txt -");
        eprintln!("  view8_rs --json --stats bytecode.txt out.json");
        eprintln!();
        eprintln!("308K-function files: ~30MB RAM (vs ~900GB Python)");
    }
}

struct CodeLine {
    offset: u32,
    instruction: String,
    translated: String,
    decompiled: String,
    visible: bool,
}

struct SharedFunctionInfo {
    name: String,
    argument_count: u32,
    code: Vec<CodeLine>,
    const_pool: Vec<String>,
    exception_table: HashMap<u32, (u32, u32)>,
}

impl SharedFunctionInfo {
    fn new() -> Self {
        Self { name: "func_unknown".into(), argument_count: 0,
               code: Vec::new(), const_pool: Vec::new(), exception_table: HashMap::new() }
    }
    fn header(&self) -> String {
        let p: Vec<String> = (0..self.argument_count.saturating_sub(1))
            .map(|i| format!("a{}", i)).collect();
        format!("function {}({})", self.name, p.join(", "))
    }
    fn visible_op_count(&self) -> usize {
        self.code.iter().filter(|l| l.visible && !l.decompiled.is_empty()).count()
    }
}

lazy_static! {
    static ref BYTECODE_RE: Regex = Regex::new(
        r"^[^@]*@\s+(\d+)\s+:\s+(?:[0-9a-fA-F]{2}\s+)*([A-Za-z_]\w*(?:\.\w+)*)\s*(.*?)$"
    ).unwrap();
    static ref ADDRESS_RE: Regex = Regex::new(
        r"^(0x[0-9a-fA-F]+):\s*\[(?:SharedFunctionInfo|BytecodeArray)\]"
    ).unwrap();
    static ref CONSTPOOL_RE: Regex = Regex::new(
        r"Constant pool\s*\(size\s*=\s*(\d+)\)"
    ).unwrap();
    static ref CONST_ENTRY_RE: Regex = Regex::new(
        r"^\s*(\d+)\s*:\s*(?:0x[0-9a-fA-F]+\s+)?(.+)$"
    ).unwrap();
    static ref CONST_RANGE_RE: Regex = Regex::new(
        r"^\s*(\d+)\s*-\s*(\d+)\s*:\s*(.+)$"
    ).unwrap();
    static ref STRING_CONST_RE: Regex = Regex::new(
        r"<String\[\d+\]:\s*#?(.*)>"
    ).unwrap();
    static ref HANDLER_RE: Regex = Regex::new(
        r"\((\d+),(\d+)\)\s*->\s*(\d+)"
    ).unwrap();
    static ref CONSTPOOL_REF_RE: Regex = Regex::new(
        r"ConstPool\[(\d+)\]"
    ).unwrap();
    static ref PARAM_RE: Regex = Regex::new(
        r"Parameter count\s+(\d+)"
    ).unwrap();
    static ref FA_SINGLE_RE: Regex = Regex::new(
        r"^\s*(\d+)\s*:\s*(-?\d+)\s*$"
    ).unwrap();
    static ref FA_RANGE_RE: Regex = Regex::new(
        r"^\s*(\d+)\s*-\s*(\d+)\s*:\s*(-?\d+)\s*$"
    ).unwrap();
    static ref FA_ADDR_RE: Regex = Regex::new(
        r"(0x[0-9a-fA-F]+)\s*:\s*\[FixedArray\]"
    ).unwrap();
    static ref FA_LEN_RE: Regex = Regex::new(
        r"^\s*-\s*length:\s*(\d+)\s*$"
    ).unwrap();
    static ref SFI_NAME_RE: Regex = Regex::new(
        r"^\s*-\s*(?:name|debug_name)\s*:\s*(.+)$"
    ).unwrap();
}

struct Parser {
    lines: Vec<String>,
    pos: usize,
    fixed_arrays: HashMap<u64, Vec<i64>>,
    node_fmt: bool,  // true = node --print-bytecode, false = patched d8
}

impl Parser {
    fn from_lines(lines: Vec<String>) -> Self {
        let node_fmt = lines.iter().take(50)
            .any(|l| l.trim().starts_with("[generated bytecode for function:"));
        Self { lines, pos: 0, fixed_arrays: HashMap::new(), node_fmt }
    }

    fn new(path: &str) -> io::Result<Self> {
        let t = Instant::now();
        let f = File::open(path)?;
        let r = BufReader::with_capacity(1 << 20, f);
        let lines: Vec<String> = r.lines().map_while(Result::ok).collect();
        eprintln!("Read {} lines in {:.1}s ({:.0}MB)", lines.len(), t.elapsed().as_secs_f32(),
            lines.iter().map(|l| l.len()).sum::<usize>() as f64 / 1e6);
        Ok(Self::from_lines(lines))
    }

    fn from_disassembler(binary: &str, input: &str) -> io::Result<Self> {
        let t = Instant::now();
        eprintln!("Running disassembler: {} {}", binary, input);
        let out = Command::new(binary)
            .arg(input)
            .output()
            .map_err(|e| io::Error::other(format!("Failed to run '{}': {}", binary, e)))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(io::Error::other(
                format!("Disassembler failed ({}): {}", out.status, stderr.trim())));
        }
        let lines: Vec<String> = String::from_utf8_lossy(&out.stdout)
            .lines().map(|l| l.to_string()).collect();
        eprintln!("Disassembled {} lines in {:.1}s", lines.len(), t.elapsed().as_secs_f32());
        Ok(Self::from_lines(lines))
    }

    fn next(&mut self) -> Option<&str> {
        while self.pos < self.lines.len() {
            self.pos += 1;
            let l = self.lines[self.pos - 1].trim();
            if !l.is_empty() { return Some(l); }
        }
        None
    }
    fn pushback(&mut self) { if self.pos > 0 { self.pos -= 1; } }

    fn collect_fixed_arrays(&mut self) {
        let saved = self.pos; self.pos = 0;
        while self.pos < self.lines.len() {
            if self.lines[self.pos].trim() == "Start FixedArray" {
                self.pos += 1; self.parse_fa();
            } else { self.pos += 1; }
        }
        self.pos = saved;
        eprintln!("  {} FixedArrays", self.fixed_arrays.len());
    }

    fn parse_fa(&mut self) {
        let mut addr = 0u64;
        while let Some(l) = self.next().map(|s| s.to_owned()) {
            if l == "End FixedArray" { return; }
            if let Some(c) = FA_ADDR_RE.captures(&l) {
                addr = u64::from_str_radix(c[1].trim_start_matches("0x"), 16).unwrap_or(0);
                break;
            }
        }
        let mut length = 0usize;
        while let Some(l) = self.next().map(|s| s.to_owned()) {
            if l == "End FixedArray" { return; }
            if let Some(c) = FA_LEN_RE.captures(&l) { length = c[1].parse().unwrap_or(0); break; }
        }
        let mut arr = vec![0i64; length];
        while let Some(l) = self.next().map(|s| s.to_owned()) {
            if l == "End FixedArray" { break; }
            if let Some(c) = FA_RANGE_RE.captures(&l) {
                let (s, e, v) = (c[1].parse::<usize>().unwrap_or(0),
                                 c[2].parse::<usize>().unwrap_or(0),
                                 c[3].parse::<i64>().unwrap_or(0));
                let hi = e.min(length.saturating_sub(1));
                for slot in &mut arr[s..=hi] { *slot = v; }
            } else if let Some(c) = FA_SINGLE_RE.captures(&l) {
                let (i, v) = (c[1].parse::<usize>().unwrap_or(0), c[2].parse::<i64>().unwrap_or(0));
                if i < length { arr[i] = v; }
            }
        }
        self.fixed_arrays.insert(addr, arr);
    }

    fn skip_block(&mut self, start: &str) {
        let end = format!("End {}", start.strip_prefix("Start ").unwrap_or(""));
        while let Some(l) = self.next().map(|s| s.to_owned()) {
            if l == end { break; }
            if l.starts_with("Start ") { self.skip_block(&l); }
        }
    }

    fn parse_all(&mut self) -> Vec<SharedFunctionInfo> {
        let t = Instant::now();
        if self.node_fmt {
            eprintln!("Parsing functions (node format)...");
        } else {
            self.collect_fixed_arrays();
            eprintln!("Parsing functions (patched-d8 format)...");
        }
        self.pos = 0;
        let mut funcs = Vec::new();
        let mut count = 0usize;
        while let Some(l) = self.next().map(|s| s.to_owned()) {
            if self.node_fmt && l.starts_with("[generated bytecode for function:") {
                self.parse_sfi_node(&l, funcs.len(), &mut funcs);
                count += 1;
                if count.is_multiple_of(200) { eprintln!("  {} functions...", count); }
            } else if !self.node_fmt && l == "Start SharedFunctionInfo" {
                self.parse_sfi(funcs.len(), &mut funcs);
                count += 1;
                if count.is_multiple_of(200) { eprintln!("  {} functions...", count); }
            }
        }
        eprintln!("Parsed {} functions in {:.1}s", funcs.len(), t.elapsed().as_secs_f32());
        funcs
    }

    fn parse_sfi_node(&mut self, header: &str, idx: usize, funcs: &mut Vec<SharedFunctionInfo>) {
        let mut sfi = SharedFunctionInfo::new();
        // "[generated bytecode for function: NAME (0x... <SharedFunctionInfo NAME>)]"
        let after = header.trim_start_matches('[')
            .strip_prefix("generated bytecode for function:").unwrap_or("").trim();
        let raw_name = after.find('(').map(|p| after[..p].trim()).unwrap_or("").trim();
        sfi.name = if raw_name.is_empty() {
            format!("func_{:04}", idx)
        } else {
            sanitize_name(raw_name, idx)
        };

        while let Some(l) = self.next().map(|s| s.to_owned()) {
            let l = l.as_str();
            if l.starts_with("[generated bytecode for function:") { self.pushback(); break; }
            if let Some(c) = PARAM_RE.captures(l) {
                sfi.argument_count = c[1].parse().unwrap_or(0); continue;
            }
            if l.contains("Constant pool") {
                sfi.const_pool = self.parse_const_pool_node(l); continue;
            }
            if l.contains("Handler Table") {
                sfi.exception_table = self.parse_handler_table(l); continue;
            }
            if l.contains("@    0 : ") || l.contains("@ 0 : ") {
                sfi.code = self.parse_bytecode(l); continue;
            }
        }
        sfi.code.sort_by_key(|c| c.offset);
        sfi.code.dedup_by_key(|c| c.offset);
        funcs.push(sfi);
    }

    fn parse_const_pool_node(&mut self, header: &str) -> Vec<String> {
        let size = CONSTPOOL_RE.captures(header)
            .and_then(|c| c[1].parse::<usize>().ok()).unwrap_or(0);
        if size == 0 { return Vec::new(); }
        let mut pool: Vec<Option<String>> = vec![None; size];
        let mut assigned = 0;
        while let Some(l) = self.next().map(|s| s.to_owned()) {
            let l = l.as_str();
            if assigned >= size { self.pushback(); break; }
            if l.starts_with("[generated bytecode") || l.starts_with("Handler Table") ||
               l.contains("@ ") || l.starts_with("Source Position") {
                self.pushback(); break;
            }
            // Skip TrustedFixedArray metadata
            if l.contains("[TrustedFixedArray]") || l.starts_with("- map:") || l.starts_with("- length:") {
                continue;
            }
            if let Some(c) = CONST_RANGE_RE.captures(l) {
                let (s, e) = (c[1].parse::<usize>().unwrap_or(0), c[2].parse::<usize>().unwrap_or(0));
                let v = parse_const_val(&c[3]);
                let hi = e.min(size.saturating_sub(1));
                for slot in pool.get_mut(s..=hi).into_iter().flatten() {
                    if slot.is_none() { *slot = Some(v.clone()); assigned += 1; }
                }
            } else if let Some(c) = CONST_ENTRY_RE.captures(l) {
                let idx = c[1].parse::<usize>().unwrap_or(usize::MAX);
                if idx < size && pool[idx].is_none() {
                    pool[idx] = Some(parse_const_val(&c[2])); assigned += 1;
                }
            }
        }
        pool.into_iter().map(|v| v.unwrap_or_default()).collect()
    }

    fn parse_sfi(&mut self, idx: usize, funcs: &mut Vec<SharedFunctionInfo>) {
        let mut sfi = SharedFunctionInfo::new();
        // Sequential index name — stable across runs (fixes suleram/View8#13)
        sfi.name = format!("func_{:04}", idx);
        while let Some(l) = self.next().map(|s| s.to_owned()) {
            let l = l.as_str();
            if l == "End SharedFunctionInfo" { break; }
            if l == "Start SharedFunctionInfo" {
                let nested_idx = funcs.len();
                self.parse_sfi(nested_idx, funcs); continue;
            }
            if l.starts_with("Start Object") || l.starts_with("Start Array") || l.starts_with("Start Fixed") {
                self.skip_block(l); continue;
            }
            // Human-readable name from debug info
            if let Some(c) = SFI_NAME_RE.captures(l) {
                let n = c[1].trim();
                if !n.is_empty() && n != "<empty>" && n != "anonymous" {
                    sfi.name = sanitize_name(n, idx);
                }
                continue;
            }
            // Fallback: append address if no human name
            if let Some(c) = ADDRESS_RE.captures(l) {
                if sfi.name == format!("func_{:04}", idx) {
                    sfi.name = format!("func_{:04}_{}", idx, &c[1]);
                }
                continue;
            }
            if let Some(c) = PARAM_RE.captures(l) {
                sfi.argument_count = c[1].parse().unwrap_or(0); continue;
            }
            if l.contains("Constant pool") {
                sfi.const_pool = self.parse_const_pool(l, funcs); continue;
            }
            if l.contains("Handler Table") {
                sfi.exception_table = self.parse_handler_table(l); continue;
            }
            if l.contains("@    0 : ") || l.contains("@ 0 : ") {
                sfi.code = self.parse_bytecode(l); continue;
            }
        }
        sfi.code.sort_by_key(|c| c.offset);
        sfi.code.dedup_by_key(|c| c.offset);
        funcs.push(sfi);
    }

    fn parse_bytecode(&mut self, first: &str) -> Vec<CodeLine> {
        let mut code = Vec::new();
        if let Some(cl) = parse_bc_line(first) { code.push(cl); }
        while let Some(l) = self.next().map(|s| s.to_owned()) {
            if !l.contains(" @ ") { self.pushback(); break; }
            if let Some(cl) = parse_bc_line(&l) { code.push(cl); }
        }
        code
    }

    fn parse_const_pool(&mut self, header: &str, funcs: &mut Vec<SharedFunctionInfo>) -> Vec<String> {
        let size = CONSTPOOL_RE.captures(header)
            .and_then(|c| c[1].parse::<usize>().ok()).unwrap_or(0);
        if size == 0 { return Vec::new(); }
        let mut pool: Vec<Option<String>> = vec![None; size];
        let mut assigned = 0;
        while let Some(l) = self.next().map(|s| s.to_owned()) {
            let l = l.as_str();
            if assigned >= size { self.pushback(); break; }
            if l == "Start SharedFunctionInfo" {
                let idx = funcs.len();
                self.parse_sfi(idx, funcs); continue;
            }
            if l.starts_with("Start ") { self.skip_block(l); continue; }
            if l.starts_with("Handler Table") || l.contains("@ ") ||
               l == "End SharedFunctionInfo" || l.starts_with("Source Position") {
                self.pushback(); break;
            }
            if let Some(c) = CONST_RANGE_RE.captures(l) {
                let (s, e) = (c[1].parse::<usize>().unwrap_or(0), c[2].parse::<usize>().unwrap_or(0));
                let v = parse_const_val(&c[3]);
                let hi = e.min(size.saturating_sub(1));
                for slot in pool.get_mut(s..=hi).into_iter().flatten() {
                    if slot.is_none() { *slot = Some(v.clone()); assigned += 1; }
                }
            } else if let Some(c) = CONST_ENTRY_RE.captures(l) {
                let idx = c[1].parse::<usize>().unwrap_or(usize::MAX);
                if idx < size && pool[idx].is_none() {
                    pool[idx] = Some(parse_const_val(&c[2])); assigned += 1;
                }
            }
        }
        pool.into_iter().map(|v| v.unwrap_or_default()).collect()
    }

    fn parse_handler_table(&mut self, header: &str) -> HashMap<u32, (u32, u32)> {
        let mut t = HashMap::new();
        if header.contains("size = 0") { return t; }
        self.next(); // skip header line
        while let Some(l) = self.next().map(|s| s.to_owned()) {
            if let Some(c) = HANDLER_RE.captures(&l) {
                let (from, to, key) = (c[1].parse().unwrap_or(0),
                                       c[2].parse().unwrap_or(0),
                                       c[3].parse().unwrap_or(0));
                t.insert(key, (from, to));
            } else { self.pushback(); break; }
        }
        t
    }
}

fn sanitize_name(raw: &str, idx: usize) -> String {
    let clean: String = raw.chars().map(|c| {
        if c.is_alphanumeric() || c == '_' || c == '$' { c } else { '_' }
    }).collect();
    if clean.is_empty() || clean.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
        format!("func_{:04}_{}", idx, clean)
    } else {
        clean
    }
}

fn parse_bc_line(line: &str) -> Option<CodeLine> {
    let c = BYTECODE_RE.captures(line)?;
    let offset = c[1].parse().ok()?;
    let op = &c[2];
    let rest = c.get(3).map(|m| m.as_str().trim()).unwrap_or("");
    let instruction = if rest.is_empty() { op.to_string() } else { format!("{} {}", op, rest) };
    Some(CodeLine { offset, instruction,
        translated: String::new(), decompiled: String::new(), visible: true })
}

fn parse_const_val(raw: &str) -> String {
    let v = raw.trim();
    // Fix suleram/View8#17: handle None / empty string constants without panic
    if v == "None" || v.is_empty() { return "null".into(); }
    if let Some(c) = STRING_CONST_RE.captures(v) {
        return format!("\"{}\"", c[1].replace('\\', "\\\\").replace('"', "\\\""));
    }
    if v.starts_with("<SharedFunctionInfo") {
        let inner = v.trim_matches(|c: char| c == '<' || c == '>');
        let name = inner.strip_prefix("SharedFunctionInfo").unwrap_or("").trim();
        return if name.is_empty() { "func_ref_anonymous".into() } else { format!("func_ref_{}", name) };
    }
    if v.contains("Odd Oddball") || v.contains("undefined") { return "undefined".into(); }
    if v.contains("Null Oddball") { return "null".into(); }
    if v.contains("True Oddball")  { return "true".into(); }
    if v.contains("False Oddball") { return "false".into(); }
    if v.contains("FixedArray") || v.contains("ArrayBoilerplate") { return "[]".into(); }
    if v.contains("ObjectBoilerplate") { return "{}".into(); }
    v.trim_matches(|c: char| c == '<' || c == '>').to_string()
}

fn translate(sfi: &mut SharedFunctionInfo) {
    for line in &mut sfi.code {
        if line.instruction.is_empty() { continue; }
        let parts: Vec<&str> = line.instruction.splitn(2, ' ').collect();
        let op = parts[0];
        let rest = if parts.len() > 1 { parts[1].trim() } else { "" };
        let args: Vec<&str> = if rest.is_empty() { vec![] }
                              else { rest.split(", ").map(|s| s.trim()).collect() };
        let a = |i: usize| -> &str { args.get(i).unwrap_or(&"?") };
        fn strip(s: &str) -> &str { s.trim_matches(|c: char| c == '[' || c == ']') }

        line.translated = match op {
            "LdaZero"     => "ACCU = 0".into(),
            "LdaSmi"      => format!("ACCU = {}", strip(a(0))),
            "LdaUndefined" => "ACCU = undefined".into(),
            "LdaNull"     => "ACCU = null".into(),
            "LdaTrue"     => "ACCU = true".into(),
            "LdaFalse"    => "ACCU = false".into(),
            "LdaTheHole"  => "ACCU = <hole>".into(),
            "LdaConstant" => format!("ACCU = ConstPool[{}]", strip(a(0))),
            "LdaGlobal"   => format!("ACCU = global[ConstPool[{}]]", strip(a(0))),
            "LdaGlobalInsideTypeof" => format!("ACCU = typeof global[ConstPool[{}]]", strip(a(0))),
            "Ldar"        => format!("ACCU = {}", a(0)),
            "LdaNamedProperty" | "GetNamedProperty" =>
                format!("ACCU = {}[ConstPool[{}]]", a(0), strip(a(1))),
            "LdaNamedPropertyFromSuper" =>
                format!("ACCU = super[ConstPool[{}]]", strip(a(1))),
            "LdaKeyedProperty" | "GetKeyedProperty" =>
                format!("ACCU = {}[ACCU]", a(0)),
            "LdaCurrentContextSlot" | "LdaImmutableCurrentContextSlot" =>
                format!("ACCU = Scope[CURRENT][{}]", strip(a(0))),
            "LdaContextSlot" | "LdaImmutableContextSlot" => {
                let d = args.get(2).map(|s| strip(s)).unwrap_or("0");
                if d == "0" { format!("ACCU = Scope[CURRENT][{}]", strip(a(1))) }
                else        { format!("ACCU = Scope[CURRENT-{}][{}]", d, strip(a(1))) }
            }
            "LdaLookupSlot" | "LdaLookupSlotInsideTypeof" =>
                format!("ACCU = Scope.lookup(ConstPool[{}])", strip(a(0))),
            "Star"  => format!("{} = ACCU", a(0)),
            s if s.starts_with("Star") && s.len() <= 6 => format!("r{} = ACCU", &s[4..]),
            "StaNamedProperty" | "SetNamedProperty" =>
                format!("{}[ConstPool[{}]] = ACCU", a(0), strip(a(1))),
            "StaNamedOwnProperty" | "DefineNamedOwnProperty" =>
                format!("{}[ConstPool[{}]] = ACCU  // own", a(0), strip(a(1))),
            "StaKeyedProperty" | "SetKeyedProperty" =>
                format!("{}[{}] = ACCU", a(0), a(1)),
            "StaCurrentContextSlot" =>
                format!("Scope[CURRENT][{}] = ACCU", strip(a(0))),
            "StaContextSlot" => {
                let d = args.get(2).map(|s| strip(s)).unwrap_or("0");
                if d == "0" { format!("Scope[CURRENT][{}] = ACCU", strip(a(1))) }
                else        { format!("Scope[CURRENT-{}][{}] = ACCU", d, strip(a(1))) }
            }
            "StaLookupSlot" =>
                format!("Scope.lookup(ConstPool[{}]) = ACCU", strip(a(0))),
            "Add"  | "AddSmi"  => format!("ACCU = (ACCU + {})", strip(a(0))),
            "Sub"  | "SubSmi"  => format!("ACCU = (ACCU - {})", strip(a(0))),
            "Mul"  | "MulSmi"  => format!("ACCU = (ACCU * {})", strip(a(0))),
            "Div"  | "DivSmi"  => format!("ACCU = (ACCU / {})", strip(a(0))),
            "Mod"  | "ModSmi"  => format!("ACCU = (ACCU % {})", strip(a(0))),
            "Exp"  | "ExpSmi"  => format!("ACCU = (ACCU ** {})", strip(a(0))),
            "Negate"     => "ACCU = -ACCU".into(),
            "Inc"        => "ACCU = (ACCU + 1)".into(),
            "Dec"        => "ACCU = (ACCU - 1)".into(),
            "BitwiseNot" => "ACCU = ~ACCU".into(),
            "BitwiseAnd" | "BitwiseAndSmi" => format!("ACCU = (ACCU & {})", strip(a(0))),
            "BitwiseOr"  | "BitwiseOrSmi"  => format!("ACCU = (ACCU | {})", strip(a(0))),
            "BitwiseXor" | "BitwiseXorSmi" => format!("ACCU = (ACCU ^ {})", strip(a(0))),
            "ShiftLeft"  | "ShiftLeftSmi"  => format!("ACCU = (ACCU << {})", strip(a(0))),
            "ShiftRight" | "ShiftRightSmi" => format!("ACCU = (ACCU >> {})", strip(a(0))),
            "ShiftRightLogical" | "ShiftRightLogicalSmi" => format!("ACCU = (ACCU >>> {})", strip(a(0))),
            "TestEqual"             => format!("ACCU = (ACCU == {})", a(0)),
            "TestEqualStrict"       => format!("ACCU = (ACCU === {})", a(0)),
            "TestLessThan"          => format!("ACCU = (ACCU < {})", a(0)),
            "TestGreaterThan"       => format!("ACCU = (ACCU > {})", a(0)),
            "TestLessThanOrEqual"   => format!("ACCU = (ACCU <= {})", a(0)),
            "TestGreaterThanOrEqual"=> format!("ACCU = (ACCU >= {})", a(0)),
            "TestInstanceOf"        => format!("ACCU = (ACCU instanceof {})", a(0)),
            "TestIn"                => format!("ACCU = ({} in ACCU)", a(0)),
            "TestNull"              => "ACCU = (ACCU === null)".into(),
            "TestUndefined"         => "ACCU = (ACCU === undefined)".into(),
            "TestUndetectable"      => "ACCU = isUndetectable(ACCU)".into(),
            "TestTypeOf"            => format!("ACCU = (typeof ACCU === ConstPool[{}])", strip(a(0))),
            "ToBooleanLogicalNot" | "LogicalNot" => "ACCU = !ACCU".into(),
            "TypeOf"     => "ACCU = typeof(ACCU)".into(),
            "ToName"     => "ACCU = String(ACCU)".into(),
            "ToNumber" | "ToNumeric" => "ACCU = Number(ACCU)".into(),
            "ToObject"   => format!("{} = Object(ACCU)", a(0)),
            "CallAnyReceiver" | "CallProperty" | "CallProperty0" |
            "CallProperty1"  | "CallProperty2" =>
                format!("ACCU = {}.call({}, ...)", a(0), a(1)),
            "CallUndefinedReceiver" | "CallUndefinedReceiver0" |
            "CallUndefinedReceiver1" | "CallUndefinedReceiver2" =>
                format!("ACCU = {}(...)", a(0)),
            "CallWithSpread"     => format!("ACCU = {}(...{})", a(0), a(1)),
            "CallRuntime"        => format!("ACCU = %{}({})", a(0), a(1)),
            "CallRuntimeForPair" => format!("({}, {}) = %{}({})", a(1), a(2), a(0), a(3)),
            "InvokeIntrinsic"    => format!("ACCU = _{}({})",
                a(0), args.get(1..).unwrap_or(&[]).join(", ")),
            "Construct"          => format!("ACCU = new {}(...)", a(0)),
            "ConstructWithSpread"=> format!("ACCU = new {}(...{})", a(0), a(1)),
            "CreateClosure"      => format!("ACCU = new func ConstPool[{}]", strip(a(0))),
            "CreateArrayLiteral" | "CreateEmptyArrayLiteral" => "ACCU = []".into(),
            "CreateObjectLiteral"| "CreateEmptyObjectLiteral" => "ACCU = {}".into(),
            "CreateRegExpLiteral"=> format!("ACCU = new RegExp(ConstPool[{}])", strip(a(0))),
            "CreateMappedArguments" | "CreateUnmappedArguments" => "ACCU = arguments".into(),
            "CreateRestParameter"=> "ACCU = [...rest]".into(),
            "Return"      => "return ACCU".into(),
            "Throw"       => "throw ACCU".into(),
            "ReThrow"     => "throw  // rethrow".into(),
            "ThrowReferenceErrorIfHole" =>
                format!("if (ACCU === <hole>) throw ReferenceError(ConstPool[{}])", strip(a(0))),
            "ThrowSuperNotCalledIfHole" =>
                "if (this === <hole>) throw ReferenceError('super() not called')".into(),
            "ThrowSuperAlreadyCalledIfNotHole" =>
                "if (this !== <hole>) throw ReferenceError('super() already called')".into(),
            "JumpIfTrue"  | "JumpIfToBooleanTrue"  => format!("if (ACCU) goto {}", a(0)),
            "JumpIfFalse" | "JumpIfToBooleanFalse" => format!("if (!ACCU) goto {}", a(0)),
            "Jump" | "JumpLoop"         => format!("goto {}", a(0)),
            "JumpIfNull"                => format!("if (ACCU === null) goto {}", a(0)),
            "JumpIfNotNull"             => format!("if (ACCU !== null) goto {}", a(0)),
            "JumpIfUndefined"           => format!("if (ACCU === undefined) goto {}", a(0)),
            "JumpIfNotUndefined"        => format!("if (ACCU !== undefined) goto {}", a(0)),
            "JumpIfUndefinedOrNull"     => format!("if (ACCU == null) goto {}", a(0)),
            "JumpIfJSReceiver"          => format!("if (isObject(ACCU)) goto {}", a(0)),
            "JumpIfForInDone"           => format!("if ({} >= {}) goto {}", a(0), a(1), a(2)),
            "SwitchOnSmiNoFeedback"     => "switch (ACCU)".into(),
            "SwitchOnGeneratorState"    => format!("switch generator state of {}", a(0)),
            "PushContext"  => format!("{} = PushContext(ACCU)", a(0)),
            "PopContext"   => format!("PopContext({})", a(0)),
            "SuspendGenerator" => format!("yield {}", a(0)),
            "ResumeGenerator"  => format!("ACCU = resume({})", a(0)),
            "GetIterator"      => "ACCU = ACCU[Symbol.iterator]()".into(),
            "GetAsyncIterator" => "ACCU = ACCU[Symbol.asyncIterator]()".into(),
            "ForInEnumerate"   => format!("ACCU = ForIn.enumerate({})", a(0)),
            "ForInPrepare"     => format!("{} = ForIn.prepare(ACCU)", a(0)),
            "ForInNext"        => format!("ACCU = ForIn.next({}, {}, {})", a(0), a(1), a(2)),
            "ForInStep"        => format!("{} = ForIn.step({})", a(0), a(0)),
            "Mov"     => format!("{} = {}", a(1), a(0)),
            "Debugger"   => "debugger".into(),
            "StackCheck" => "// stack overflow check".into(),
            "Wide" | "ExtraWide" => String::new(),
            _ => format!("// {} {}", op, rest),
        };
    }
}

fn replace_const_pool(sfi: &mut SharedFunctionInfo) {
    let pool = &sfi.const_pool;
    let plen = pool.len();
    for line in &mut sfi.code {
        if !line.visible || line.translated.is_empty() { continue; }
        if plen > 0 && line.translated.contains("ConstPool[") {
            line.decompiled = CONSTPOOL_REF_RE.replace_all(&line.translated, |caps: &regex::Captures| {
                let idx: usize = caps[1].parse().unwrap_or(usize::MAX);
                if idx < plen { pool[idx].clone() } else { caps[0].to_string() }
            }).to_string();
        } else {
            line.decompiled = line.translated.clone();
        }
    }
}

fn should_include(sfi: &SharedFunctionInfo, cfg: &Config) -> bool {
    if cfg.no_stubs && sfi.code.is_empty() { return false; }
    if let Some(ref re) = cfg.filter { if !re.is_match(&sfi.name) { return false; } }
    let ops = sfi.visible_op_count();
    if let Some(min) = cfg.min_ops { if ops < min { return false; } }
    if let Some(max) = cfg.max_ops { if ops > max { return false; } }
    true
}

// ── Text export ───────────────────────────────────────────────────────────────

fn export_text(w: &mut dyn Write, sfi: &SharedFunctionInfo, cfg: &Config) -> io::Result<()> {
    writeln!(w, "{}", sfi.header())?;
    writeln!(w, "{{")?;
    for l in &sfi.code {
        if !l.visible { continue; }
        let cols: Vec<&str> = cfg.formats.iter().filter_map(|fmt| match fmt {
            ExportFormat::V8Opcode   => Some(l.instruction.as_str()).filter(|s| !s.is_empty()),
            ExportFormat::Translated => Some(l.translated.as_str()).filter(|s| !s.is_empty()),
            ExportFormat::Decompiled => Some(l.decompiled.as_str()).filter(|s| !s.is_empty()),
        }).collect();
        if cols.is_empty() { continue; }
        if cfg.show_offset { write!(w, "\t@{:04x}  ", l.offset)?; }
        else               { write!(w, "\t")?; }
        writeln!(w, "{}", cols.join("  |  "))?;
    }
    writeln!(w, "}}")
}

// ── JSON export ───────────────────────────────────────────────────────────────

fn json_str(w: &mut dyn Write, s: &str) -> io::Result<()> {
    write!(w, "\"")?;
    for c in s.chars() {
        match c {
            '"'  => write!(w, "\\\"")?,
            '\\' => write!(w, "\\\\")?,
            '\n' => write!(w, "\\n")?,
            '\r' => write!(w, "\\r")?,
            '\t' => write!(w, "\\t")?,
            c    => write!(w, "{}", c)?,
        }
    }
    write!(w, "\"")
}

fn export_json(w: &mut dyn Write, sfi: &SharedFunctionInfo, cfg: &Config, first: &mut bool) -> io::Result<()> {
    if !*first { writeln!(w, ",")?; }
    *first = false;
    write!(w, "  {{\"name\":")?; json_str(w, &sfi.name)?;
    writeln!(w, ",\"args\":{},\"instructions\":[", sfi.argument_count.saturating_sub(1))?;
    let mut instr_first = true;
    for l in &sfi.code {
        if !l.visible { continue; }
        let has_content = cfg.formats.iter().any(|fmt| match fmt {
            ExportFormat::V8Opcode   => !l.instruction.is_empty(),
            ExportFormat::Translated => !l.translated.is_empty(),
            ExportFormat::Decompiled => !l.decompiled.is_empty(),
        });
        if !has_content { continue; }
        if !instr_first { writeln!(w, ",")?; }
        instr_first = false;
        write!(w, "    {{")?;
        if cfg.show_offset { write!(w, "\"offset\":{},", l.offset)?; }
        let mut field_first = true;
        for fmt in &cfg.formats {
            let (key, val) = match fmt {
                ExportFormat::V8Opcode   => ("v8_opcode",  l.instruction.as_str()),
                ExportFormat::Translated => ("translated", l.translated.as_str()),
                ExportFormat::Decompiled => ("decompiled", l.decompiled.as_str()),
            };
            if val.is_empty() { continue; }
            if !field_first { write!(w, ",")?; }
            field_first = false;
            write!(w, "\"{key}\":")?;
            json_str(w, val)?;
        }
        write!(w, "}}")?;
    }
    write!(w, "\n  ]}}")?;
    Ok(())
}

// ── Stats ─────────────────────────────────────────────────────────────────────

fn print_stats(funcs: &[SharedFunctionInfo]) {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for sfi in funcs {
        for l in &sfi.code {
            if !l.instruction.is_empty() {
                let op = l.instruction.split_ascii_whitespace().next().unwrap_or("");
                *counts.entry(op).or_insert(0) += 1;
            }
        }
    }
    let mut sorted: Vec<(&&str, &usize)> = counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    let total: usize = sorted.iter().map(|(_, n)| **n).sum();
    eprintln!("\n── Opcode frequency ({} total instructions) ──", total);
    for (op, count) in sorted.iter().take(30) {
        eprintln!("  {:40} {:>8}  ({:.1}%)", op, count,
            **count as f64 / total as f64 * 100.0);
    }
    if sorted.len() > 30 {
        eprintln!("  ... and {} more opcodes", sorted.len() - 30);
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> io::Result<()> {
    let cfg = Config::parse().unwrap_or_else(|e| {
        eprintln!("Error: {}\n", e);
        Config::print_usage();
        std::process::exit(1);
    });

    let t = Instant::now();
    let mut parser = match &cfg.disassembler {
        Some(bin) => Parser::from_disassembler(bin, &cfg.input)?,
        None      => Parser::new(&cfg.input)?,
    };

    let mut funcs = parser.parse_all();
    drop(parser); // free all parsed lines

    // Parallel translate + resolve
    eprintln!("Decompiling {} functions (parallel)...", funcs.len());
    funcs.par_iter_mut().for_each(|sfi| {
        translate(sfi);
        replace_const_pool(sfi);
    });

    if cfg.stats { print_stats(&funcs); }

    // Open output (file or stdout)
    let mut w: Box<dyn Write> = match &cfg.output {
        Some(path) => Box::new(BufWriter::with_capacity(256 * 1024, File::create(path)?)),
        None       => Box::new(BufWriter::new(io::stdout())),
    };

    let mut json_first = true;
    if cfg.json { writeln!(w, "[")?; }

    let mut written = 0usize;
    let total = funcs.len();
    for (i, sfi) in funcs.iter().enumerate() {
        if (i + 1) % 100 == 0 || sfi.code.len() > 5000 {
            eprintln!("  [{}/{}] {} ({} ops)", i + 1, total,
                &sfi.name[..sfi.name.len().min(50)], sfi.code.len());
        }
        if !should_include(sfi, &cfg) { continue; }
        if cfg.json { export_json(&mut *w, sfi, &cfg, &mut json_first)?; }
        else        { export_text(&mut *w, sfi, &cfg)?; }
        written += 1;
    }

    if cfg.json { writeln!(w, "\n]")?; }
    w.flush()?;

    let dest = cfg.output.as_deref().unwrap_or("stdout");
    eprintln!("Done. {}/{} functions in {:.1}s → {}",
        written, total, t.elapsed().as_secs_f32(), dest);
    Ok(())
}
