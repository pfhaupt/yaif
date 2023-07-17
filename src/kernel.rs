#![allow(dead_code)]

use opencl3::event::Event;
use opencl3::command_queue::CommandQueue;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::Buffer;
use opencl3::Result;


const MMUL_VERSION: usize = 3;

pub const TS: usize = 32;

const WPT: usize = 4;
const RTS: usize = TS / WPT;

const TSM: usize = 8;
const TSN: usize = TSM;
const TSK: usize = 8;
const WPTM: usize = 4;
const WPTN: usize = WPTM;
const RTSM: usize = TSM / WPTM;
const RTSN: usize = TSN / WPTN;
const LPTA: usize = (TSK * TSM) / (RTSM * RTSN);
const LPTB: usize = (TSK * TSN) / (RTSM * RTSN);


pub const ADD_NAME: &str = "matrix_add";
pub const ADD_SOURCE: &str = r#"
kernel void matrix_add(
    const int M,
    const int N,
    global float const* A,
    global float const* B,
    global float* C)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int i = globalCol * M + globalRow;
    C[i] = A[i] + B[i];
}"#;


pub const MUL_SCALAR_NAME: &str = "matrix_mul_scalar";
pub const MUL_SCALAR_SOURCE: &str = r#"
kernel void matrix_mul_scalar(
    const int M,
    const int N,
    global float* x,
    float y)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    const int i = globalCol * M + globalRow;
    x[i] = x[i] * y;
}"#;

// Thanks a lot to https://cnugteren.github.io/tutorial/pages/page1.html for the awesome tutorial.
pub const MUL_MATRIX_NAME: &str = "matrix_mul_matrix";
pub const MUL_MATRIX_SOURCE: &str =
if MMUL_VERSION == 1 {
r#"
kernel void matrix_mul_matrix(
    const int M,
    const int N,
    const int K,
    global float const* A,
    global float const* B,
    global float* C)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[k * M + globalRow] * B[globalCol * K + k];
    }
    C[globalCol * M + globalRow] = acc;
}
"#
} else if MMUL_VERSION == 2 { // Version 2 (working)
r#"
#define TS 32
kernel void matrix_mul_matrix(
    const int M,
    const int N,
    const int K,
    global float const* A,
    global float const* B,
    global float* C,
    local float* Asub,
    local float* Bsub)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = TS * get_group_id(0) + row;
    const int globalCol = TS * get_group_id(1) + col;

    float acc = 0.0f;
    
    const int numTiles = K/TS;
    for (int t = 0; t < numTiles; t++) {
        const int tiledRow = TS * t + row;
        const int tiledCol = TS * t + col;
        Asub[col * TS + row] = A[tiledCol * M + globalRow];
        Bsub[col * TS + row] = B[globalCol * K + tiledRow];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k=0; k<TS; k++) {
            acc += Asub[k * TS + row] * Bsub[col * TS + k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[globalCol * M + globalRow] = acc;
}
"#
} else if MMUL_VERSION == 3 { // Version 3 (working)
r#"
#define TS 32
#define WPT 4
#define RTS (TS / WPT)
kernel void matrix_mul_matrix(
    const int M,
    const int N,
    const int K,
    global float const* A,
    global float const* B,
    global float* C,
    local float* Asub,
    local float* Bsub)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = TS * get_group_id(0) + row;
    const int globalCol = TS * get_group_id(1) + col;

    float acc[WPT];
    for (int w = 0; w < WPT; w++) {
        acc[w] = 0.0f;
    }

    const int numTiles = K / TS;
    for (int t = 0; t < numTiles; t++) {
        for (int w = 0; w < WPT; w++) {
            const int tiledRow = TS * t + row;
            const int tiledCol = TS * t + col;
            Asub[(col + w * RTS) * TS + row] = A[(tiledCol + w * RTS) * M + globalRow];
            Bsub[(col + w * RTS) * TS + row] = B[(globalCol + w * RTS) * K + tiledRow];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            for (int w = 0; w < WPT; w++) {
                acc[w] += Asub[k * TS + row] * Bsub[(col + w * RTS) * TS + k];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WPT; w++) {
        C[(globalCol + w * RTS) * M + globalRow] = acc[w];
    }
}
"#
} else if MMUL_VERSION == 4 { // Version 4 (not working anymore :( )
r#"
#define MOD2(x,y) ((x) % (y))
#define DIV2(x,y) ((x) / (y))
#define TS 32
#define WPT 4
#define RTS (TS / WPT)
#define TSM 8
#define TSN TSM
#define TSK 8
#define WPTM 4
#define WPTN WPTM
#define RTSM (TSM / WPTM)
#define RTSN (TSN / WPTN)
#define LPTA ((TSK * TSM)/(RTSM * RTSN))
#define LPTB ((TSK * TSN)/(RTSM * RTSN))
kernel void matrix_mul_matrix(
    const int M,
    const int N,
    const int K,
    global float const* A,
    global float const* B,
    global float* C,
    local float* Asub,
    local float* Bsub)
{
    const int tidm = get_local_id(0);
    const int tidn = get_local_id(1);
    const int offsetM = TSM * get_group_id(0);
    const int offsetN = TSN * get_group_id(1);

    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    #pragma unroll
    for (int wm=0; wm < WPTM; wm++) {
        #pragma unroll
        for (int wn=0; wn < WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    const int numTiles = K/TSK;
    int t = 0;
    do {
        #pragma unroll
        for (int la = 0; la < LPTA; la++) {
            int tid = tidn * RTSM + tidm;
            volatile int id = la * RTSN * RTSM + tid;
            int row = MOD2(id, TSM);
            int col = DIV2(id, TSM);
            int tiledIndex = TSK * t + col;
            Asub[col * TSK + row] = A[tiledIndex * M + offsetM + row];
            Bsub[row * TSN + col] = B[tiledIndex * N + offsetN + row];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TSK; k++) {
            #pragma unroll
            for (int wn = 0; wn < WPTN; wn++) {
                int col = tidn + wn * RTSN;
                Breg[wn] = Bsub[col * TSN + k];
            }
            #pragma unroll
            for (int wm = 0; wm < WPTM; wm++) {
                int row = tidm + wm * RTSM;
                Areg = Asub[k * TSK + row];
                #pragma unroll
                for (int wn = 0; wn < WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        t++;
    } while (t < numTiles);

    #pragma unroll
    for (int wm = 0; wm < WPTM; wm++) {
        int globalRow = offsetM + tidm + wm * RTSM;
        #pragma unroll
        for (int wn = 0; wn < WPTN; wn++) {
            int globalCol = offsetN + tidn + wn * RTSN;
            C[globalCol * M + globalRow] = acc[wm][wn];
        }
    }
}
"#
} else {
    ""
};



pub fn get_mmul_kernel_event(kernel: &Kernel, queue: &CommandQueue, m: usize, n: usize, k: usize, a: &Buffer<f32>, b: &Buffer<f32>, c: &Buffer<f32>) -> Result<Event> {
    let mut exec_kernel = ExecuteKernel::new(&kernel);
    let event = exec_kernel
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(&(k as u32))
        .set_arg(a)
        .set_arg(b)
        .set_arg(c);
    let result = match MMUL_VERSION {
        1 => {
            event.set_global_work_sizes(&[m, n, 1])
                .set_local_work_sizes(&[TS, TS, 1])
                .enqueue_nd_range(&queue)?
        },
        2 => {
            event.set_arg_local_buffer(TS * TS * 4)
                .set_arg_local_buffer(TS * TS * 4)
                .set_global_work_sizes(&[m, n, 1])
                .set_local_work_sizes(&[TS, TS, 1])
                .enqueue_nd_range(&queue)?
        },
        3 => {
            event.set_arg_local_buffer(TS * TS * 4)
                .set_arg_local_buffer(TS * TS * 4)
                .set_global_work_sizes(&[m, n / WPT, 1])
                .set_local_work_sizes(&[TS, TS / WPT, 1])
                .enqueue_nd_range(&queue)?
        },
        4 => {
            event.set_arg_local_buffer(TSK * TSM * 4)
                .set_arg_local_buffer(TSN * (TSK + 2) * 4)
                .set_global_work_sizes(&[m / WPTM, n / WPTN, 1])
                .set_local_work_sizes(&[RTSM, RTSN, 1])
                .enqueue_nd_range(&queue)?
        },
        _ => {
            panic!("Invalid MMUL_VERSION selected! {}", MMUL_VERSION);
        }
    };
    Ok(result)
}

pub fn get_madd_kernel_event(kernel: &Kernel, queue: &CommandQueue, m: usize, n: usize, a: &Buffer<f32>, b: &Buffer<f32>, c: &Buffer<f32>) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(a)
        .set_arg(b)
        .set_arg(c)
        .set_global_work_sizes(&[m, n, 1])
        .set_local_work_sizes(&[TS, TS, 1])
        .enqueue_nd_range(&queue)
}

pub fn get_smul_kernel_event(kernel: &Kernel, queue: &CommandQueue, m: usize, n: usize, a: &Buffer<f32>, scalar: f32) -> Result<Event> {
    ExecuteKernel::new(&kernel)
        .set_arg(&(m as u32))
        .set_arg(&(n as u32))
        .set_arg(a)
        .set_arg(&scalar)
        .set_global_work_sizes(&[m, n, 1])
        .set_local_work_sizes(&[TS, TS, 1])
        .enqueue_nd_range(&queue)
}