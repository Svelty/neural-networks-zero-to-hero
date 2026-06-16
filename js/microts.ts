class Value {
  data: number;
  grad: number;
  children: Value[];
  op: string;
  label: string;
  _backward: () => void;

  constructor(data: number, children: Value[], op: string, label: string = "") {
    this.data = data;
    this.grad = 0;
    this.children = children;
    this.op = op;
    this.label = label;
    this._backward = () => {};
  }

  add(other: Value): Value {
    const out = new Value(this.data + other.data, [this, other], "+");
    out._backward = () => {
      this.grad += 1 * out.grad;
      other.grad += 1 * out.grad;
    };
    return out;
  }

  multipy(other: Value): Value {
    const out = new Value(this.data * other.data, [this, other], "*");
    out._backward = () => {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    };
    return out;
  }

  // minus

  // multiply

  // power

  // divide

  // negative

  // tahn

  // exp

  // backward
}
