use yaif::data_set::{self, DataSet};
use yaif::net::cl_net::ClNet;
use yaif::net::NetTrait;
use yaif::cl_kernel::ClStruct;

fn main() {
    let inputs = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let outputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0, 0.0]];
    let data_set = DataSet::new(&inputs, &outputs);
    let cl_struct = ClStruct::new().unwrap();
    match ClNet::new(cl_struct, vec![2, 1, 2]) {
        Ok(mut cl) => {
            cl.initialize_training_data(&data_set);
            cl.run(true);
            println!("Worked!!")
        },
        Err(e) => {
            panic!("{}", e)
        }
    }
}