# Jeras
Jeras is a machine learning framework for Java. The motivation for this repository is to better understand the underlying code of neural networks. This is an experimental framework designed to work just like Keras and is just for educational purposes.   
   
---   
   
## Usage
To use Jeras you can either create an instance of a network
```Java
int inputs = x.columns;
int hidden1 = 8;
int hidden2 = 8;
int outputs = y.columns;
int epochs = 1000;
float lr = 0.1f;

int[] networkShape = {inputs, hidden1, hidden2, outputs};

MLP nn = new MLP(networkShape, lr);

nn.train(x, y, epochs);
System.out.println(nn.predict(x));
```
   
Or use the Sequential syntax just like Keras  
```Java
int epochs = 5000;
float lr = 0.1f;

Sequential model = new Sequential();
model.add(new Dense(8, "sigmoid", x.columns));
model.add(new Dense(8, "sigmoid"));
model.add(new Dense(y.columns, "softmax"));
model.compile(lr);

model.fit(x, y, epochs);
System.out.println(model.predict(x));
```

So far I only have Dense layers but plan to implement Convolusional layers next.

