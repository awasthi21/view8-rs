# view8-rs

A fast V8 bytecode decompiler written in Rust. Parses the text output of `--print-bytecode` and reconstructs a pseudo-JavaScript representation of each function.

## Why Rust?

The original [view8](https://github.com/nicowillis/view8) Python tool works well but struggles with large V8 dump files. For a file with ~308K bytecode lines:

| Tool | Peak RAM |
|------|----------|
| Python view8 | ~900 GB (GC + dicts + string copies) |
| **view8-rs** | **~30 MB** (flat structs, no GC) |

## Install

```bash
cargo install --path view8-rs
```

Or build manually:

```bash
git clone https://github.com/awasthi21/view8-rs
cd view8-rs/view8-rs
cargo build --release
# binary at: target/release/view8_rs
```

## Usage

First, generate a V8 bytecode dump:

```bash
node --print-bytecode your_script.js > bytecode.txt 2>&1
```

Then decompile it:

```bash
view8_rs bytecode.txt output.js
```

### Example output

```js
function func_start_0x7f1a2b3c(a0, a1) {
    ACCU = a0
    r0 = ACCU
    ACCU = r0["length"]
    if (!ACCU) goto +12
    ACCU = r0.call(r1, ...)
    return ACCU
}
```

## How it works

1. **Parse** — reads the entire dump into memory line-by-line, resolving `FixedArray` and `SharedFunctionInfo` blocks
2. **Translate** — maps each V8 opcode to a readable pseudo-JS expression (accumulator model)
3. **Resolve** — replaces `ConstPool[N]` references with their actual string/value from the constant pool
4. **Export** — writes one function block per `SharedFunctionInfo` to the output file

## Dependencies

- [`regex`](https://crates.io/crates/regex) — bytecode line parsing
- [`lazy_static`](https://crates.io/crates/lazy_static) — compiled regex pool
- [`memmap2`](https://crates.io/crates/memmap2) — fast file I/O

## License

MIT
