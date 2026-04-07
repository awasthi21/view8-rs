use lazy_static::lazy_static;
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
    output: String,
    disassembler: Option<String>,
    formats: Vec<ExportFormat>,
}

impl Config {
    fn parse() -> Result<Self, String> {
        let args: Vec<String> = env::args().collect();
        let mut input = None;
        let mut output = None;
        let mut disassembler = None;
        let mut formats = Vec::new();
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
                            other => return Err(format!("Unknown format '{}'. Valid: v8_opcode, translated, decompiled", other)),
                        });
                        i += 1;
                    }
                    continue;
                }
                _ if !args[i].starts_with('-') => {
                    if input.is_none() { input = Some(args[i].clone()); }
                    else if output.is_none() { output = Some(args[i].clone()); }
                    else { return Err(format!("Unexpected argument: {}", args[i])); }
                }
                other => return Err(format!("Unknown option: {}", other)),
            }
            i += 1;
        }
        if formats.is_empty() { formats.push(ExportFormat::Decompiled); }
        Ok(Config {
            input:  input.ok_or("Missing input file")?,
            output: output.ok_or("Missing output file")?,
            disassembler,
            formats,
        })
    }

    fn print_usage() {
        eprintln!("view8_rs — V8 bytecode decompiler (Rust)");
        eprintln!("Usage: view8_rs [options] <input> <output>");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  -p, --path <binary>           Path to disassembler binary (e.g. d8)");
        eprintln!("                                 If omitted, input must be pre-disassembled text");
        eprintln!("  -e, --export-format <fmt...>  Output format(s) (default: decompiled)");
        eprintln!("                                   v8_opcode   raw V8 opcode + operands");
        eprintln!("                                   translated  accumulator-model pseudo-JS");
        eprintln!("                                   decompiled  resolved constants (default)");
        eprintln!("                                 Multiple formats are written as columns:");
        eprintln!("                                   -e v8_opcode translated decompiled");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  # pre-disassembled input (node --print-bytecode output)");
        eprintln!("  view8_rs bytecode.txt out.js");
        eprintln!();
        eprintln!("  # raw .jsc file via disassembler binary");
        eprintln!("  view8_rs -p ./d8 input.jsc out.js");
        eprintln!();
        eprintln!("  # multiple export formats side by side");
        eprintln!("  view8_rs -e v8_opcode decompiled bytecode.txt out.js");
        eprintln!();
        eprintln!("  # all three formats");
        eprintln!("  view8_rs -e v8_opcode translated decompiled bytecode.txt out.js");
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
        let p: Vec<String> = (0..self.argument_count.saturating_sub(1)).map(|i| format!("a{}", i)).collect();
        format!("function {}({})", self.name, p.join(", "))
    }
}

lazy_static! {
    static ref BYTECODE_RE: Regex = Regex::new(r"^[^@]*@\s+(\d+)\s+:\s+(?:[0-9a-fA-F]{2}\s+)*([A-Za-z_]\w*(?:\.\w+)*)\s*(.*?)$").unwrap();
    static ref ADDRESS_RE: Regex = Regex::new(r"^(0x[0-9a-fA-F]+):\s*\[(?:SharedFunctionInfo|BytecodeArray)\]").unwrap();
    static ref CONSTPOOL_RE: Regex = Regex::new(r"Constant pool\s*\(size\s*=\s*(\d+)\)").unwrap();
    static ref CONST_ENTRY_RE: Regex = Regex::new(r"^\s*(\d+)\s*:\s*(?:0x[0-9a-fA-F]+\s+)?(.+)$").unwrap();
    static ref CONST_RANGE_RE: Regex = Regex::new(r"^\s*(\d+)\s*-\s*(\d+)\s*:\s*(.+)$").unwrap();
    static ref STRING_CONST_RE: Regex = Regex::new(r"<String\[\d+\]:\s*#?(.+?)>").unwrap();
    static ref HANDLER_RE: Regex = Regex::new(r"\((\d+),(\d+)\)\s*->\s*(\d+)").unwrap();
    static ref CONSTPOOL_REF_RE: Regex = Regex::new(r"ConstPool\[(\d+)\]").unwrap();
    static ref PARAM_RE: Regex = Regex::new(r"Parameter count\s+(\d+)").unwrap();
    static ref FA_SINGLE_RE: Regex = Regex::new(r"^\s*(\d+)\s*:\s*(-?\d+)\s*$").unwrap();
    static ref FA_RANGE_RE: Regex = Regex::new(r"^\s*(\d+)\s*-\s*(\d+)\s*:\s*(-?\d+)\s*$").unwrap();
    static ref FA_ADDR_RE: Regex = Regex::new(r"(0x[0-9a-fA-F]+)\s*:\s*\[FixedArray\]").unwrap();
    static ref FA_LEN_RE: Regex = Regex::new(r"^\s*-\s*length:\s*(\d+)\s*$").unwrap();
}

struct Parser {
    lines: Vec<String>,
    pos: usize,
    fixed_arrays: HashMap<u64, Vec<i64>>,
}

impl Parser {
    fn from_lines(lines: Vec<String>) -> Self {
        Self { lines, pos: 0, fixed_arrays: HashMap::new() }
    }

    fn new(path: &str) -> io::Result<Self> {
        let t = Instant::now();
        let f = File::open(path)?;
        let r = BufReader::with_capacity(1 << 20, f);
        let lines: Vec<String> = r.lines().filter_map(|l| l.ok()).collect();
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
            .map_err(|e| io::Error::new(io::ErrorKind::Other,
                format!("Failed to run '{}': {}", binary, e)))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(io::Error::new(io::ErrorKind::Other,
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
        loop { let l = match self.next() { Some(s) => s.to_string(), None => break }; let l = l.as_str();
            if l == "End FixedArray" { return; }
            if let Some(c) = FA_ADDR_RE.captures(l) {
                addr = u64::from_str_radix(c[1].trim_start_matches("0x"), 16).unwrap_or(0);
                break;
            }
        }
        let mut length = 0usize;
        loop { let l = match self.next() { Some(s) => s.to_string(), None => break }; let l = l.as_str();
            if l == "End FixedArray" { return; }
            if let Some(c) = FA_LEN_RE.captures(l) { length = c[1].parse().unwrap_or(0); break; }
        }
        let mut arr = vec![0i64; length];
        loop { let l = match self.next() { Some(s) => s.to_string(), None => break }; let l = l.as_str();
            if l == "End FixedArray" { break; }
            if let Some(c) = FA_RANGE_RE.captures(l) {
                let (s,e,v) = (c[1].parse::<usize>().unwrap_or(0), c[2].parse::<usize>().unwrap_or(0), c[3].parse::<i64>().unwrap_or(0));
                for i in s..=e.min(length.saturating_sub(1)) { arr[i] = v; }
            } else if let Some(c) = FA_SINGLE_RE.captures(l) {
                let (i,v) = (c[1].parse::<usize>().unwrap_or(0), c[2].parse::<i64>().unwrap_or(0));
                if i < length { arr[i] = v; }
            }
        }
        self.fixed_arrays.insert(addr, arr);
    }

    fn skip_block(&mut self, start: &str) {
        let kind = start.strip_prefix("Start ").unwrap_or("");
        let end = format!("End {}", kind);
        loop { let l = match self.next() { Some(s) => s.to_string(), None => break }; let l = l.as_str();
            if l == end { break; }
            if l.starts_with("Start ") { self.skip_block(l); }
        }
    }

    fn parse_all(&mut self) -> Vec<SharedFunctionInfo> {
        let t = Instant::now();
        self.collect_fixed_arrays();
        eprintln!("Parsing functions...");
        self.pos = 0;
        let mut funcs = Vec::new();
        let mut count = 0usize;
        loop { let l = match self.next() { Some(s) => s.to_string(), None => break }; let l = l.as_str();
            if l == "Start SharedFunctionInfo" {
                self.parse_sfi("start", &mut funcs);
                count += 1;
                if count % 200 == 0 { eprintln!("  {} functions...", count); }
            }
        }
        eprintln!("Parsed {} functions in {:.1}s", funcs.len(), t.elapsed().as_secs_f32());
        funcs
    }

    fn parse_sfi(&mut self, hint: &str, funcs: &mut Vec<SharedFunctionInfo>) {
        let mut sfi = SharedFunctionInfo::new();
        loop { let l = match self.next() { Some(s) => s.to_string(), None => break }; let l = l.as_str();
            if l == "End SharedFunctionInfo" { break; }
            if l == "Start SharedFunctionInfo" {
                let n = format!("nested_{}", funcs.len());
                self.parse_sfi(&n, funcs); continue;
            }
            if l.starts_with("Start Object") || l.starts_with("Start Array") || l.starts_with("Start Fixed") {
                self.skip_block(l); continue;
            }
            if let Some(c) = ADDRESS_RE.captures(l) {
                sfi.name = format!("func_{}_{}", hint, &c[1]); continue;
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
        loop { let l = match self.next() { Some(s) => s.to_string(), None => break }; let l = l.as_str();
            if !l.contains(" @ ") { self.pushback(); break; }
            if let Some(cl) = parse_bc_line(l) { code.push(cl); }
        }
        code
    }

    fn parse_const_pool(&mut self, header: &str, funcs: &mut Vec<SharedFunctionInfo>) -> Vec<String> {
        let size = CONSTPOOL_RE.captures(header).and_then(|c| c[1].parse::<usize>().ok()).unwrap_or(0);
        if size == 0 { return Vec::new(); }
        let mut pool: Vec<Option<String>> = vec![None; size];
        let mut assigned = 0;
        loop { let l = match self.next() { Some(s) => s.to_string(), None => break }; let l = l.as_str();
            if assigned >= size { self.pushback(); break; }
            if l == "Start SharedFunctionInfo" {
                let n = format!("nested_cp_{}", funcs.len());
                self.parse_sfi(&n, funcs); continue;
            }
            if l.starts_with("Start ") { self.skip_block(l); continue; }
            if l.starts_with("Handler Table") || l.contains("@ ") || l == "End SharedFunctionInfo" || l.starts_with("Source Position") {
                self.pushback(); break;
            }
            if let Some(c) = CONST_RANGE_RE.captures(l) {
                let (s,e) = (c[1].parse::<usize>().unwrap_or(0), c[2].parse::<usize>().unwrap_or(0));
                let v = parse_const_val(&c[3]);
                for i in s..=e { if i < size && pool[i].is_none() { pool[i] = Some(v.clone()); assigned += 1; } }
            } else if let Some(c) = CONST_ENTRY_RE.captures(l) {
                let idx = c[1].parse::<usize>().unwrap_or(usize::MAX);
                if idx < size && pool[idx].is_none() { pool[idx] = Some(parse_const_val(&c[2])); assigned += 1; }
            }
        }
        pool.into_iter().map(|v| v.unwrap_or_default()).collect()
    }

    fn parse_handler_table(&mut self, header: &str) -> HashMap<u32, (u32, u32)> {
        let mut t = HashMap::new();
        if header.contains("size = 0") { return t; }
        self.next(); // skip header line
        loop { let l = match self.next() { Some(s) => s.to_string(), None => break }; let l = l.as_str();
            if let Some(c) = HANDLER_RE.captures(l) {
                let (from,to,key) = (c[1].parse().unwrap_or(0), c[2].parse().unwrap_or(0), c[3].parse().unwrap_or(0));
                t.insert(key, (from, to));
            } else { self.pushback(); break; }
        }
        t
    }
}

fn parse_bc_line(line: &str) -> Option<CodeLine> {
    let c = BYTECODE_RE.captures(line)?;
    let offset = c[1].parse().ok()?;
    let op = &c[2];
    let rest = c.get(3).map(|m| m.as_str().trim()).unwrap_or("");
    Some(CodeLine { offset, instruction: format!("{} {}", op, rest).trim_end().to_string(),
        translated: String::new(), decompiled: String::new(), visible: true })
}

fn parse_const_val(raw: &str) -> String {
    let v = raw.trim();
    if let Some(c) = STRING_CONST_RE.captures(v) { return format!("\"{}\"", c[1].replace('"', "\\\"")); }
    if v.starts_with("<SharedFunctionInfo") { return format!("func_ref_{}", v.split_whitespace().last().unwrap_or("?").trim_end_matches('>')); }
    if v.contains("Odd Oddball") { return "null".into(); }
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
        let args: Vec<&str> = if rest.is_empty() { vec![] } else { rest.split(", ").map(|s| s.trim()).collect() };
        let a = |i: usize| -> &str { args.get(i).unwrap_or(&"?") };
        fn strip(s: &str) -> &str { s.trim_matches(|c: char| c == '[' || c == ']') }

        line.translated = match op {
            "LdaZero" => "ACCU = 0".into(),
            "LdaSmi" => format!("ACCU = {}", strip(a(0))),
            "LdaUndefined" => "ACCU = undefined".into(),
            "LdaNull" => "ACCU = null".into(),
            "LdaTrue" => "ACCU = true".into(),
            "LdaFalse" => "ACCU = false".into(),
            "LdaConstant" => format!("ACCU = ConstPool[{}]", strip(a(0))),
            "LdaGlobal" => format!("ACCU = global[ConstPool[{}]]", strip(a(0))),
            "Ldar" => format!("ACCU = {}", a(0)),
            "LdaNamedProperty" | "GetNamedProperty" => format!("ACCU = {}[ConstPool[{}]]", a(0), strip(a(1))),
            "LdaKeyedProperty" | "GetKeyedProperty" => format!("ACCU = {}[ACCU]", a(0)),
            "LdaCurrentContextSlot" | "LdaImmutableCurrentContextSlot" => format!("ACCU = Scope[CURRENT][{}]", strip(a(0))),
            "LdaContextSlot" | "LdaImmutableContextSlot" => {
                let d = args.get(2).map(|s| strip(s)).unwrap_or("0");
                if d == "0" { format!("ACCU = Scope[CURRENT][{}]", strip(a(1))) }
                else { format!("ACCU = Scope[CURRENT-{}][{}]", d, strip(a(1))) }
            }
            "Star" => format!("{} = ACCU", a(0)),
            s if s.starts_with("Star") && s.len() <= 6 => format!("r{} = ACCU", &s[4..]),
            "StaNamedProperty" | "SetNamedProperty" => format!("{}[ConstPool[{}]] = ACCU", a(0), strip(a(1))),
            "StaKeyedProperty" | "SetKeyedProperty" => format!("{}[{}] = ACCU", a(0), a(1)),
            "StaCurrentContextSlot" => format!("Scope[CURRENT][{}] = ACCU", strip(a(0))),
            "StaContextSlot" => {
                let d = args.get(2).map(|s| strip(s)).unwrap_or("0");
                if d == "0" { format!("Scope[CURRENT][{}] = ACCU", strip(a(1))) }
                else { format!("Scope[CURRENT-{}][{}] = ACCU", d, strip(a(1))) }
            }
            "Add" | "AddSmi" => format!("ACCU = (ACCU + {})", strip(a(0))),
            "Sub" => format!("ACCU = (ACCU - {})", a(0)),
            "Mul" => format!("ACCU = (ACCU * {})", a(0)),
            "Div" => format!("ACCU = (ACCU / {})", a(0)),
            "Mod" => format!("ACCU = (ACCU % {})", a(0)),
            "Negate" => "ACCU = -ACCU".into(),
            "Inc" => "ACCU = (ACCU + 1)".into(),
            "Dec" => "ACCU = (ACCU - 1)".into(),
            "BitwiseNot" => "ACCU = ~ACCU".into(),
            "TestEqual" => format!("ACCU = (ACCU == {})", a(0)),
            "TestEqualStrict" => format!("ACCU = (ACCU === {})", a(0)),
            "TestLessThan" => format!("ACCU = (ACCU < {})", a(0)),
            "TestGreaterThan" => format!("ACCU = (ACCU > {})", a(0)),
            "TestInstanceOf" => format!("ACCU = (ACCU instanceof {})", a(0)),
            "TestNull" => "ACCU = (ACCU === null)".into(),
            "TestUndefined" => "ACCU = (ACCU === undefined)".into(),
            "ToBooleanLogicalNot" | "LogicalNot" => "ACCU = !ACCU".into(),
            "TypeOf" => "ACCU = typeof(ACCU)".into(),
            "CallAnyReceiver" | "CallProperty" | "CallProperty0" | "CallProperty1" | "CallProperty2" =>
                format!("ACCU = {}.call({}, ...)", a(0), a(1)),
            "CallUndefinedReceiver" | "CallUndefinedReceiver0" | "CallUndefinedReceiver1" | "CallUndefinedReceiver2" =>
                format!("ACCU = {}(...)", a(0)),
            "CallRuntime" => format!("ACCU = %{}({})", a(0), a(1)),
            "InvokeIntrinsic" => format!("ACCU = _{}({})", a(0), args.get(1..).unwrap_or(&[]).join(", ")),
            "Construct" => format!("ACCU = new {}(...)", a(0)),
            "CreateClosure" => format!("ACCU = new func ConstPool[{}]", strip(a(0))),
            "CreateArrayLiteral" | "CreateEmptyArrayLiteral" => "ACCU = new []".into(),
            "CreateObjectLiteral" | "CreateEmptyObjectLiteral" => "ACCU = new {}".into(),
            "CreateRegExpLiteral" => format!("ACCU = new RegExp(ConstPool[{}])", strip(a(0))),
            "Return" => "return ACCU".into(),
            "Throw" => "throw ACCU".into(),
            "JumpIfTrue" | "JumpIfToBooleanTrue" => format!("if (ACCU) goto {}", a(0)),
            "JumpIfFalse" | "JumpIfToBooleanFalse" => format!("if (!ACCU) goto {}", a(0)),
            "Jump" | "JumpLoop" => format!("goto {}", a(0)),
            "JumpIfNull" => format!("if (ACCU === null) goto {}", a(0)),
            "JumpIfUndefined" => format!("if (ACCU === undefined) goto {}", a(0)),
            "JumpIfUndefinedOrNull" => format!("if (ACCU == null) goto {}", a(0)),
            "SwitchOnSmiNoFeedback" => "switch (ACCU)".into(),
            "PushContext" => format!("{} = PushContext(ACCU)", a(0)),
            "PopContext" => format!("PopContext({})", a(0)),
            "SuspendGenerator" => format!("yield {}", a(0)),
            "ResumeGenerator" => format!("ACCU = resume({})", a(0)),
            "GetIterator" => "ACCU = ACCU[Symbol.iterator]()".into(),
            "Mov" => format!("{} = {}", a(1), a(0)),
            "Debugger" => "debugger".into(),
            "Wide" | "ExtraWide" => String::new(),
            _ => format!("// {} {}", op, rest),
        };
    }
}

fn replace_const_pool(sfi: &mut SharedFunctionInfo) {
    if sfi.const_pool.is_empty() { return; }
    let pool = &sfi.const_pool;
    let plen = pool.len();
    for line in &mut sfi.code {
        if !line.visible || line.translated.is_empty() { continue; }
        if line.translated.contains("ConstPool[") {
            line.decompiled = CONSTPOOL_REF_RE.replace_all(&line.translated, |caps: &regex::Captures| {
                let idx: usize = caps[1].parse().unwrap_or(usize::MAX);
                if idx < plen { pool[idx].clone() } else { caps[0].to_string() }
            }).to_string();
        } else {
            line.decompiled = line.translated.clone();
        }
    }
}

fn export(w: &mut BufWriter<File>, sfi: &SharedFunctionInfo, formats: &[ExportFormat]) -> io::Result<()> {
    writeln!(w, "{}", sfi.header())?;
    writeln!(w, "{{")?;
    for l in &sfi.code {
        if !l.visible { continue; }
        let cols: Vec<&str> = formats.iter().filter_map(|fmt| match fmt {
            ExportFormat::V8Opcode   => Some(l.instruction.as_str()).filter(|s| !s.is_empty()),
            ExportFormat::Translated => Some(l.translated.as_str()).filter(|s| !s.is_empty()),
            ExportFormat::Decompiled => Some(l.decompiled.as_str()).filter(|s| !s.is_empty()),
        }).collect();
        if !cols.is_empty() {
            writeln!(w, "\t{}", cols.join("  |  "))?;
        }
    }
    writeln!(w, "}}")
}

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

    let total = funcs.len();
    eprintln!("Decompiling {} functions...", total);

    let f = File::create(&cfg.output)?;
    let mut w = BufWriter::with_capacity(256 * 1024, f);

    for (i, mut sfi) in funcs.drain(..).enumerate() {
        if (i+1) % 100 == 0 || sfi.code.len() > 5000 {
            eprintln!("  [{}/{}] {} ({} ops)", i+1, total, &sfi.name[..sfi.name.len().min(50)], sfi.code.len());
        }
        translate(&mut sfi);
        replace_const_pool(&mut sfi);
        export(&mut w, &sfi, &cfg.formats)?;
        // sfi DROPPED here — memory freed instantly, no GC
    }

    w.flush()?;
    eprintln!("Done. {} functions in {:.1}s → {}", total, t.elapsed().as_secs_f32(), &cfg.output);
    Ok(())
}
