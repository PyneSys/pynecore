//! SHA-1 (FIPS 180-4), for the one digest a step bakes into its output: the
//! Security step's per-module sec id prefix (`hashlib.sha1(path).hexdigest()[:8]`).
//!
//! Written out rather than taken from a crate: it is a few dozen lines with no
//! platform dependence, so the WebAssembly build needs nothing extra, and it is
//! gated against `hashlib` by the Rust tests and by every sec id the parity
//! tests compare.

/// The SHA-1 digest of `data`.
pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut state: [u32; 5] = [0x6745_2301, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476, 0xC3D2_E1F0];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());
    for block in message.as_chunks::<64>().0 {
        let mut words = [0u32; 80];
        for (index, word) in block.as_chunks::<4>().0.iter().enumerate() {
            words[index] = u32::from_be_bytes(*word);
        }
        for index in 16..80 {
            words[index] =
                (words[index - 3] ^ words[index - 8] ^ words[index - 14] ^ words[index - 16])
                    .rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = state;
        for (index, &word) in words.iter().enumerate() {
            let (f, k) = match index {
                0..=19 => ((b & c) | (!b & d), 0x5A82_7999),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let temp =
                a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e]) {
            *slot = slot.wrapping_add(value);
        }
    }
    let mut digest = [0u8; 20];
    for (chunk, word) in digest.as_chunks_mut::<4>().0.iter_mut().zip(state) {
        chunk.copy_from_slice(&word.to_be_bytes());
    }
    digest
}

/// The lowercase hex spelling of `data`'s SHA-1 digest (`hashlib.sha1(data).hexdigest()`).
pub fn sha1_hex(data: &[u8]) -> String {
    sha1(data).iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::sha1_hex;

    #[test]
    fn known_digests() {
        assert_eq!(sha1_hex(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(
            sha1_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
        assert_eq!(sha1_hex(&[b'a'; 1_000_000]), "34aa973cd4c4daa4f61eeb2bdbad27316534016f");
        assert_eq!(sha1_hex(b"<script>"), "fb1c2d66bbf94a08a5bda7f06316c022ca66cfa7");
        assert_eq!(
            sha1_hex("/Users/x/\u{e9}.py".as_bytes()),
            "7868d75d029a15eb554d04e286f5f89c568951da"
        );
        let bytes: Vec<u8> = (0..3).flat_map(|_| 0..=255u8).collect();
        assert_eq!(sha1_hex(&bytes), "ac2a264c8ec1f4232a40854e8239bc3a697ab1d2");
    }
}
