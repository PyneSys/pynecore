// Run JSON requests through the WebAssembly transform: one request per stdin
// line, one response per stdout line (see crates/pyne-transform-wasm).
//
//     node rust/tools/wasm_gate.mjs path/to/pyne_transform_wasm.wasm < requests.jsonl
import { readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';

const { instance } = await WebAssembly.instantiate(readFileSync(process.argv[2]), {});
const { memory, pyne_alloc, pyne_free, pyne_transform } = instance.exports;
const encoder = new TextEncoder();
const decoder = new TextDecoder();

for await (const line of createInterface({ input: process.stdin })) {
    const request = encoder.encode(line);
    const ptr = pyne_alloc(request.length);
    new Uint8Array(memory.buffer, ptr, request.length).set(request);
    const packed = pyne_transform(ptr, request.length);
    pyne_free(ptr, request.length);
    const outPtr = Number(packed >> 32n);
    const outLen = Number(packed & 0xffffffffn);
    process.stdout.write(decoder.decode(new Uint8Array(memory.buffer, outPtr, outLen)) + '\n');
    pyne_free(outPtr, outLen);
}
