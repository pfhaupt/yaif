#![feature(portable_simd)]

use std::simd::{f32x8, Simd};
use rand::Rng;
use std::time::Instant;

fn simd_from_vec(v: Vec<f32>) -> Vec<Simd<f32, 8>> {
    let mut r = vec![f32x8::default(); v.len() / 8];
    for i in (0..v.len()).step_by(8) {
        let range = i..(i + 8);
        r[i / 8] = f32x8::from_slice(&v[range]);
    }
    r
}

fn init_vec(size: usize) -> (Vec<f32>, Vec<f32>) {
    let mut v1= vec![0.0; size];
    let mut v2 = vec![0.0; size];
    for i in 0..size {
        let e1 = rand::thread_rng().gen_range(0..100);
        let e2 = rand::thread_rng().gen_range(0..100);
        v1[i] = e1 as f32;
        v2[i] = e2 as f32;
    }
    (v1, v2)
}

fn test_simd(tests: usize, size: usize) {
    for _ in 0..tests {
        let (v1, v2) = init_vec(size);
        let v1 = simd_from_vec(v1);
        let v2 = simd_from_vec(v2);
        let mut r = vec![];
        for i in 0..v1.len() {
            for e in (v1[i] + v2[i]).to_array() {
                r.push(e);
            }
        }
    }
}

fn test_solo(tests: usize, size: usize) {
    for _ in 0..tests {
        let (v1, v2) = init_vec(size);
        let mut r = vec![];
        for i in 0..v1.len() {
            r.push(v1[i] + v2[i]);
        }
    }
}

fn main() {
    const TESTS: usize = 10_000;
    let size = 400;
    let now = Instant::now();
    test_simd(TESTS, size);
    println!("SIMD f32 ops:  {:?}", now.elapsed());
    let now = Instant::now();
    test_solo(TESTS, size);
    println!("Basic f32 ops: {:?}", now.elapsed());
}
