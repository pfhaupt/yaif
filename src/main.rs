use yaif::data_set::{DataSet};
use yaif::net::cl_net::ClNet;
use yaif::net::NetTrait;
use yaif::net::nn::NN;
use yaif::cl_kernel::ClStruct;

fn main() {
    let inputs = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
    let outputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0, 0.0]];
    let data_set = DataSet::new(&inputs, &outputs);

    match NN::new(vec![2, 2, 2]) {
        Ok(mut n) => {
            n.initialize_training_data(&data_set);
            if n.run(false) {
                n.print_guess();
                println!("yes");
            } else {
                println!("no");
            }
        },
        Err(e) => {
            panic!("{}", e)
        }
    }
    let cl_struct = ClStruct::new().unwrap();
    match ClNet::new(cl_struct, vec![2, 2, 2]) {
        Ok(mut cl) => {
            cl.initialize_training_data(&data_set);
            if cl.run(true) {
                cl.print_guess();
                println!("yes");
            } else {
                println!("no");
            }
        },
        Err(e) => {
            panic!("{}", e)
        }
    }
}