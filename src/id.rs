use typing::Type;

pub struct IdGen {
    n: usize,
}

impl IdGen {
    pub fn new() -> IdGen {
        IdGen { n: 0 }
    }

    pub fn get_id(&mut self) -> usize {
        let n = self.n;
        self.n += 1;
        n
    }
    pub fn get_type(&mut self) -> Type {
        Type::Var(self.get_id())
    }
}
