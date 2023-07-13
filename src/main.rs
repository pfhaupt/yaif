use yaif::nn::NN;


fn main() {
    let nn = NN::new(vec![10, 3]);
    if nn.is_ok() {
        let nn = nn.unwrap();
        println!("{:?}", nn);
    }
    else {
        println!("{:?}", nn.unwrap_err());
    }
}