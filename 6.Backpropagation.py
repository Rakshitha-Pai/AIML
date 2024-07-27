import numpy as np
x=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
x=x/np.amax(x,axis=0)
y=y/100

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivaties_sigmoid(x):
    return x*(1-x)
epoch=5000
lr=0.1
inputlayer_neurons=2
hiddenlayer_neurons=3
output_neurons=1
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))

wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))

bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):
    hinp1=np.dot(x,wh)
    hinp=hinp1+bh
    hlayer_act=sigmoid(hinp)
    outinp1=np.dot(hlayer_act,wout)
    outinp=outinp1+bout
    output=sigmoid(outinp)
    

    eo=y-output
    outgrad=derivaties_sigmoid(output)
    d_output=eo*outgrad
    eh=d_output.dot(wout.T)
    
    
    hiddengrad=derivaties_sigmoid(hlayer_act)
    d_hiddenlayer=eh*hiddengrad
    
    wout+=hlayer_act.T.dot(d_output)*lr
    wh+=x.T.dot(d_hiddenlayer)*lr
    
print("Input:\n"+str(x))
print("Actual Output :\n"+str(y))
print("Predicted Output :\n",output)
    






import numpy as np
x = np.array([[2, 9], [1, 5], [3, 6]], dtype=float) / 9
y = np.array([[92], [86], [89]], dtype=float) / 100
sigmoid = lambda x: 1 / (1 + np.exp(-x))
deriv_sigmoid = lambda x: x * (1 - x)
epoch = 5000
lr = 0.1
wh, bh = np.random.uniform(size=(2, 3)), np.random.uniform(size=(1, 3))
wout, bout = np.random.uniform(size=(3, 1)), np.random.uniform(size=(1, 1))
for _ in range(epoch):
    hlayer_act = sigmoid(np.dot(x, wh) + bh)
    output = sigmoid(np.dot(hlayer_act, wout) + bout)

    d_output = (y - output) * deriv_sigmoid(output)
    d_hiddenlayer = np.dot(d_output, wout.T) * deriv_sigmoid(hlayer_act)

    wout += np.dot(hlayer_act.T, d_output) * lr
    wh += np.dot(x.T, d_hiddenlayer) * lr
predicted_output = output
print("Input (scaled):\n", x)
print("Actual Output (scaled):\n", y)
print("Predicted Output (scaled):\n", predicted_output)
