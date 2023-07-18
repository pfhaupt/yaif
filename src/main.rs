use rand::Rng;
use yaif::data_set::DataSet;
use yaif::net::cl_net::ClNet;
use yaif::net::NetTrait;
use yaif::net::nn::NN;
use yaif::cl_kernel::ClStruct;

fn main() {
    let xor_inputs = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
    let xor_outputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0, 0.0]];

    let xor_layers = vec![2, 2, 2];

    let mut identity_inputs = vec![];
    let mut identity_outputs = vec![];
    let identity_length = 100;
    for i in 1..=identity_length {
        identity_inputs.push(vec![i as f32; identity_length]);
        identity_outputs.push(vec![i as f32; identity_length]);
    }

    let mut add_inputs = vec![];
    let mut add_outputs = vec![];
    const MAX: usize = 10;
    for _ in 1..=(MAX * MAX * MAX) {
        let e1 = rand::thread_rng().gen_range(0..MAX);
        let e2 = rand::thread_rng().gen_range(0..MAX);
        let i = vec![e1 as f32, e2 as f32];
        let mut o = vec![0.0; 2 * MAX];
        o[e1 + e2] = 1.0;
        add_inputs.push(i);
        add_outputs.push(o);
    }
    let add_layers = vec![2, 64, 2 * MAX];

    let identity_layers = vec![identity_length, identity_length];

    let test = "XOR";
    let (layer_sizes, inputs, outputs) = match test { 
        "XOR" => { (xor_layers, xor_inputs, xor_outputs) },
        "ID"  => { (identity_layers, identity_inputs, identity_outputs) },
        "ADD" => { (add_layers, add_inputs, add_outputs ) }
        _ => { panic!() }
    };
    
    let data_set = DataSet::new(&inputs, &outputs);

    match NN::new(layer_sizes.clone()) {
        Ok(mut n) => {
            n.initialize_training_data(&data_set);
            match n.run(true) {
                Some(iter_count) => {
                    println!("Normal Network finished after {} iterations!", iter_count);
                    n.print_guess();
                },
                None => {
                    println!("Normal Network was not able to solve XOR");
                }
            }
        },
        Err(e) => {
            panic!("{}", e)
        }
    }
    let cl_struct = ClStruct::new().unwrap();
    match ClNet::new(cl_struct, layer_sizes.clone()) {
        Ok(mut n) => {
            n.initialize_training_data(&data_set);
            match n.run(true) {
                Some(iter_count) => {
                    println!("OpenCL Network finished after {} iterations!", iter_count);
                    n.print_guess();
                },
                None => {
                    println!("OpenCL Network was not able to solve XOR");
                }
            }
        },
        Err(e) => {
            panic!("{}", e)
        }
    }
}