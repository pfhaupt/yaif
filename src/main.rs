use yaif::nn::NN;


fn main() {
    let nn = NN::new(vec![10, 10, 3]);
    if nn.is_ok() {
        let mut nn = nn.unwrap();
        println!("{:?}", nn);
        nn.learn();
    }
    else {
        println!("{:?}", nn.unwrap_err());
    }
}