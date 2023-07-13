use yaif::Matrix;


fn main() {
    let mut m1 = Matrix::new(2, 2);
    let mut m2 = Matrix::new(2, 2);
    m1.set_unchecked(0, 0, 9.0);
    m1.set_unchecked(1, 0, 8.0);
    m1.set_unchecked(0, 1, 10.0);
    m1.set_unchecked(1, 1, 9.0);
    
    m2.set_unchecked(0, 0, 8.0);
    m2.set_unchecked(1, 0, 8.0);
    m2.set_unchecked(0, 1, 10.0);
    m2.set_unchecked(1, 1, 8.0);

    let m3 = m1.multiply(&m2).unwrap();
    println!("{:?}", m3);
}