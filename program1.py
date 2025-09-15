def mcp_neuron(inputs, weights, threshold):
    net_input=sum(i*w for i,w in zip(inputs,weights))
    return 1 if net_input >= threshold else 0

def AND_gate(x1, x2):
    weights=[1,1]
    threshold=2
    return mcp_neuron([x1,x2], weights, threshold)

def OR_gate(x1,x2):
    weights=[1,1]
    threshold=1
    return mcp_neuron([x1,x2], weights, threshold)

def NOT_gate(x):
    weights=[-1]
    threshold=0
    return mcp_neuron([x], weights, threshold)

print("AND Gate:")
for x1 in [0,1]:
    for x2 in [0,1]:
        print(f"{x1} AND {x2} = {AND_gate(x1,x2)}")

print("\nOR Gate:")
for x1 in [0,1]:
    for x2 in [0,1]:
        print(f"{x1} OR {x2} = {OR_gate(x1,x2)}")

print("\nNOT Gate:")
for x in [0,1]:
    print(f"NOT {x} = {NOT_gate(x)}")
