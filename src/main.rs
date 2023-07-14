use yaif::{nn::NN, data_set::DataSet};

fn parse_input() {
    match std::fs::File::open("mnist/train-images.idx3-ubyte") {
        Ok(file) => {
            println!("{:?}", file);
            
        },
        Err(e) => {
            panic!("{}", e);
        }
    }
}
fn mnist() {
    let layers = vec![28 * 28, 100, 10];
    // let data = parse_mnist();
    let inputs: Vec<Vec<f32>> = parse_input();
    // let outputs: Vec<Vec<f32>> = get_output(&data);
    // let data_set = DataSet::new(&inputs, &outputs);
    match NN::new(layers) {
        Ok(nn) => {
            println!("worked");
            // nn.initialize_data(&data_set);
        },
        Err(e) => {
            panic!("Could not init Network! {}", e);
        }
    }
}

fn main() {
    mnist();
}