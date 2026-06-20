class Value {
  data: number;
  grad: number;
  label: string;
  private _prev: Set<Value>;
  _op: string;
  private _backward: () => void;

  constructor(
    data: number,
    children: Value[] = [],
    op: string = "",
    label: string = "",
  ) {
    this.data = data;
    this.grad = 0;
    this._prev = new Set(children);
    this._op = op;
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

  multiply(other: Value): Value {
    const out = new Value(this.data * other.data, [this, other], "*");
    out._backward = () => {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    };
    return out;
  }

  negative(): Value {
    return this.multiply(new Value(-1));
  }

  subtract(other: Value): Value {
    return this.add(other.negative());
  }

  powerOf(x: number) {
    const out = new Value(this.data ** x, [this], `**${x}`);
    out._backward = () => {
      this.grad += x * this.data ** (x - 1) * out.grad;
    };
    return out;
  }

  divide(other: Value) {
    return this.multiply(other.powerOf(-1));
  }

  tanh() {
    const x = this.data;
    const t = (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1);
    const out = new Value(t, [this], "tanh");
    out._backward = () => {
      this.grad += (1 - t ** 2) * out.grad;
    };
    return out;
  }

  exp() {
    const x = this.data;
    const out = new Value(Math.exp(x), [this], "exp");
    out._backward = () => {
      this.grad += out.data * out.grad;
    };
    return out;
  }

  backward() {
    const topo: Value[] = [];
    const visited = new Set();

    const buildTopo = (n: Value) => {
      if (!visited.has(n)) {
        visited.add(n);
        for (const child of n._prev) {
          buildTopo(child);
        }
        topo.push(n);
      }
    };
    buildTopo(this);

    this.grad = 1;
    for (const node of topo.reverse()) {
      node._backward();
    }
  }
}

class Neuron {
  w: Value[];
  b: Value;

  constructor(nin: number) {
    // this.w = [];
    // for (let x = 0; x < nin; x++) {
    //   this.w.push(new Value(Math.random() * 2 - 1));
    // }
    this.w = Array.from(
      { length: nin },
      () => new Value(Math.random() * 2 - 1),
    );
    this.b = new Value(Math.random() * 2 - 1);
  }

  forward(x: Value[]) {
    let act = null;
    for (let i = 0; i < x.length; i++) {
      const xi = x[i];
      const wi = this.w[i];

      act == null ? (act = xi.multiply(wi)) : (act = act.add(xi.multiply(wi)));
    }
    act = act == null ? (act = this.b) : act.add(this.b);
    const out = act.tanh();
    return out;
  }

  parameters() {
    return this.w.concat([this.b]);
  }
}

class Layer {
  neurons: Neuron[];

  constructor(nin: number, nout: number) {
    this.neurons = Array.from({ length: nout }, () => new Neuron(nin));
  }

  forward(x: Value[]) {
    const outs = Array.from({ length: this.neurons.length }, (_, i) =>
      this.neurons[i].forward(x),
    );
    return outs.length == 1 ? outs[0] : outs;
  }

  parameters() {
    const params = [];
    for (let neuron of this.neurons) {
      for (let p of neuron.parameters()) {
        params.push(p);
      }
    }
    return params;
  }
}

class MLP {
  layers: Layer[];

  constructor(nin: number, nouts: number[]) {
    const sz = [nin].concat(nouts);
    this.layers = Array.from(
      { length: nouts.length },
      (_, i) => new Layer(sz[i], sz[i + 1]),
    );
  }

  forward(x: Value[]): Value {
    for (const layer of this.layers) {
      //@ts-ignore
      x = layer.forward(x);
    }
    //@ts-ignore
    return x;
  }

  parameters() {
    const params = [];
    for (let layer of this.layers) {
      for (let p of layer.parameters()) params.push(p);
    }
    return params;
  }
}

const x = [new Value(2), new Value(3), new Value(-1)];
const n = new MLP(3, [4, 4, 1]);

console.log(n);
console.log("=================");

let i = 0;
for (const layer of n.layers) {
  for (const neuron of layer.neurons) {
    for (const w of neuron.w) {
      console.log(
        `neuron ${i++} ${w.label} ${w._op}: weight: ${w.data}, grad: ${w.grad}`,
      );
    }
  }
}

const result = n.forward(x);

console.log(n);
console.log(result);
result.backward();
i = 0;
for (const layer of n.layers) {
  for (const neuron of layer.neurons) {
    for (const w of neuron.w) {
      console.log(
        `neuron ${i++} ${w.label} ${w._op}: weight: ${w.data}, grad: ${w.grad}`,
      );
    }
  }
}

const xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
];
const ys = [1.0, -1.0, -1.0, 1.0];

let ypred;
for (let i = 0; i < 20; i++) {
  ypred = xs.map((x) => n.forward(x.map((v) => new Value(v))));

  let loss = null;
  for (let j = 0; j < ypred.length; j++) {
    const ygt = ys[j];
    const yout = ypred[j];
    loss == null
      ? (loss = new Value(ygt).subtract(yout).powerOf(2))
      : (loss = loss.add(new Value(ygt).subtract(yout).powerOf(2)));
  }

  for (let p of n.parameters()) {
    p.grad = 0;
  }
  loss?.backward();

  for (let p of n.parameters()) {
    p.data += -0.05 * p.grad;
  }

  console.log(i, loss?.data);
}

console.log("finished");
console.log(ypred);
