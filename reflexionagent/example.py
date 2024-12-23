from langchain_core.runnables import RunnableLambda

#This runs in sequence
sequence = RunnableLambda(lambda x: x+1) | RunnableLambda(lambda x: x*2)
res= sequence.invoke(1)
print(res)
res =sequence.batch([1,2,3])
print(res)


#This runs in parallel
sequence = RunnableLambda(lambda x: x + 1) | {
    'mul_2': RunnableLambda(lambda x: x * 2),
    'mul_5': RunnableLambda(lambda x: x * 5)
}
print(sequence.invoke(1)) # {'mul_2': 4, 'mul_5': 10}
