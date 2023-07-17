use yaif::net::cl_net::ClNet;
use yaif::cl_kernel::ClStruct;

fn main() {
    let cl_struct = ClStruct::new().unwrap();
    match ClNet::new(cl_struct, vec![5, 5]) {
        Ok(cl) => {
            println!("Worked!!")
        },
        Err(e) => {
            panic!("{}", e)
        }
    }
}