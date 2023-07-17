use yaif::net::cl_net::ClNet;
use yaif::cl_kernel::ClStruct;

fn main() {
    let cl_struct = ClStruct::new().unwrap();
    let mut _cl_net = ClNet::new(cl_struct, vec![28, 28]);
}