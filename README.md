Below is the complete text with questions and corresponding pseudocode answers. You can use this text to generate your PDF.

────────────────────────────
**Question 1: Approaches to Writing Parallel Programs and MIMD Systems**

*Question:*  
In modern computing, performance improvements rely heavily on parallel programming techniques. Describe different approaches to writing parallel programs, providing examples of each. Additionally, explain how parallel systems are classified and differentiate them from concurrent and distributed systems. How do the principal types of MIMD systems contribute to efficient parallel computing?

*Answer (Pseudocode):*

```
// ----- Thread-Based Parallelism (Shared Memory) -----
function threadParallelExample():
    // Create a pool of threads
    for each thread in THREAD_POOL:
         // Each thread runs concurrently
         parallel_execute:
              print("Hello from thread", thread.id)

// ----- Message Passing (Distributed Memory) -----
function mpiExample():
    MPI_Init()
    rank = MPI_Comm_rank()
    if rank == 0:
         print("Hello from process", rank)
    else:
         // Other processes perform their tasks
         performLocalWork()
    MPI_Finalize()

// ----- Data Parallelism -----
function dataParallelAdd(a, b, c, n):
    parallel for i from 0 to n-1:
         c[i] = a[i] + b[i]

// ----- Task Parallelism -----
function taskParallel():
    // Execute independent tasks concurrently
    parallel_execute:
         execute(task1)
         execute(task2)
         execute(task3)

// ----- MIMD System Classification -----
// Shared-memory MIMD uses threads (e.g., OpenMP, Pthreads)
// Distributed-memory MIMD uses MPI for message passing
// Hybrid systems combine both models
```

────────────────────────────
**Question 2: Von Neumann Architecture and Memory Models**

*Question:*  
Explain the Von Neumann architecture and its significance in modern computing. How does it relate to parallel and distributed systems? Additionally, compare shared memory and distributed memory MIMD systems.

*Answer (Pseudocode):*

```
// ----- Von Neumann Execution Cycle -----
function vonNeumannCycle():
    load_program_and_data_from_memory()
    while not end_of_program:
         instruction = fetch_from_memory()
         decoded = decode(instruction)
         execute(decoded)

// Note:
// Shared Memory MIMD: multiple processors access a common memory.
// Distributed Memory MIMD: each processor has its own memory and communicates via message passing.
```

────────────────────────────
**Question 3: Cloud-Based AI Training Platform**

*Question:*  
A cloud-based AI training platform is designed to process large-scale machine learning workloads by distributing computations across multiple independent servers, each with its own private memory. The system must efficiently exchange model parameters between nodes while minimizing communication overhead. In this scenario, which type of MIMD system would be ideal, and how does it handle data exchange between processors?

*Answer (Pseudocode):*

```
// Distributed Memory MIMD is ideal.
function distributedAITraining():
    MPI_Init()
    local_data = getLocalData()
    local_update = computeLocalModelUpdate(local_data)
    // Use MPI_Allreduce to sum (or average) model updates across nodes
    global_update = MPI_Allreduce(local_update, operation=SUM)
    updateModel(global_update)
    MPI_Finalize()
```

────────────────────────────
**Question 4: Pthreads Dynamic Task Queue**

*Question:*  
You are tasked with developing a multi-threaded system using Pthreads to manage a dynamic task queue. The program should start with a user-specified number of worker threads that initially sleep in a condition wait. The main thread generates blocks of tasks and wakes a worker thread with a condition signal. After completing its tasks, the worker returns to waiting. Once all tasks are generated, the main thread signals completion and uses a condition broadcast to wake all threads for termination. Implement the system with proper synchronization.

*Answer (Pseudocode):*

```
// Global shared variables
taskQueue = empty queue
mutex = new Mutex()
conditionVar = new ConditionVariable()
tasksDone = false

// Function to enqueue a task
function enqueueTask(task):
    lock(mutex)
    taskQueue.enqueue(task)
    signal(conditionVar)  // Wake one waiting worker
    unlock(mutex)

// Worker thread function
function workerThread(threadId):
    while true:
         lock(mutex)
         while taskQueue.isEmpty() and (tasksDone == false):
              wait(conditionVar, mutex)
         if taskQueue.isEmpty() and tasksDone:
              unlock(mutex)
              break
         task = taskQueue.dequeue()
         unlock(mutex)
         process(task)         // Process the task
         sleep(1)              // Simulate work
    print("Thread", threadId, "exiting")

// Main thread
function main():
    // Create worker threads
    for i from 1 to numWorkers:
         start workerThread(i)
    // Generate tasks in blocks
    for each block in taskBlocks:
         for each task in block:
              enqueueTask(task)
         sleep(2)   // Simulate delay between blocks
    lock(mutex)
    tasksDone = true
    broadcast(conditionVar)  // Wake all waiting workers to exit
    unlock(mutex)
    wait for all worker threads to finish
```

────────────────────────────
**Question 5: Handling Imbalanced Workload in Matrix-Vector Multiplication**

*Question:*  
Imagine you are implementing a parallel matrix-vector multiplication algorithm for a large dataset. Initially, the matrix dimensions are perfectly divisible by the number of available threads. However, if the dimensions do not divide evenly, describe an approach to handle such an imbalanced workload efficiently.

*Answer (Pseudocode):*

```
// m: total rows, t: total threads
function assignRows(m, t):
    rows_per_thread = m / t   // Integer division
    remainder = m mod t
    for i from 0 to t - 1:
         start = i * rows_per_thread + min(i, remainder)
         if i < remainder:
             count = rows_per_thread + 1
         else:
             count = rows_per_thread
         end = start + count - 1
         assignTaskToThread(i, start, end)
```

────────────────────────────
**Question 6: OpenMP Histogram Program**

*Question:*  
Develop an OpenMP program to implement a histogram where the master thread receives sensor data in the range 0-99, and slave threads (each representing a bucket) count the data falling within their range. The master thread displays bucket counts periodically.

*Answer (Pseudocode):*

```
// Global variables
NUM_BUCKETS = 10
sensorData[TOTAL_DATA]
bucketCounts[NUM_BUCKETS] = {0}
dataIndex = 0  // Shared counter

// Set total threads = NUM_BUCKETS + 1 (1 master + NUM_BUCKETS workers)
parallel region shared(sensorData, bucketCounts, dataIndex):
    threadId = omp_get_thread_num()
    if threadId == 0:
         // Master thread: generate sensor data
         for i from 0 to TOTAL_DATA - 1:
              sensorData[i] = randomInteger(0, 99)
              dataIndex++
              if (i mod 100 == 0):
                  print(bucketCounts)
              sleep(delay)
    else:
         // Worker threads: each corresponds to one bucket
         bucket = threadId - 1
         lowerBound = bucket * (100 / NUM_BUCKETS)
         upperBound = lowerBound + (100 / NUM_BUCKETS)
         myIndex = 0
         while true:
              localIndex = dataIndex
              while myIndex < localIndex:
                    if sensorData[myIndex] >= lowerBound and sensorData[myIndex] < upperBound:
                         atomic_increment(bucketCounts[bucket])
                    myIndex++
              if localIndex >= TOTAL_DATA:
                    break
              sleep(shortDelay)
After parallel region:
print("Final bucket counts:", bucketCounts)
```

────────────────────────────
**Question 7: MPI Global Min/Max Shipping Times**

*Question:*  
You are running a large e-commerce operation where each warehouse computes the minimum and maximum shipping times. Using MPI, write a program flow to collect these local min/max values and compute the global minimum and maximum shipping times.

*Answer (Pseudocode):*

```
// In each process:
function aggregateShippingTimes():
    MPI_Init()
    localMin = computeLocalMinShippingTime()
    localMax = computeLocalMaxShippingTime()
    globalMin = MPI_Reduce(localMin, operation=MIN, root=0)
    globalMax = MPI_Reduce(localMax, operation=MAX, root=0)
    if (rank == 0):
         print("Global Minimum Shipping Time:", globalMin)
         print("Global Maximum Shipping Time:", globalMax)
    MPI_Finalize()
```

────────────────────────────
**Question 8: Ping-Pong Communication and Timing**

*Question:*  
You implement a ping-pong communication pattern between two processes to measure message passing cost. How long does the program need to run for clock() to report a nonzero runtime? How do timings from clock() compare to those from MPI_Wtime()?

*Answer (Pseudocode):*

```
// In process 0:
function pingPongTest():
    MPI_Init()
    if rank == 0:
         startWall = MPI_Wtime()
         startCPU = clock()
         for count from 0 to PING_PONG_LIMIT - 1:
              MPI_Send(message, destination=1)
              MPI_Recv(message, source=1)
         endCPU = clock()
         endWall = MPI_Wtime()
         cpuTime = (endCPU - startCPU) / CLOCKS_PER_SEC  // CPU time
         wallTime = endWall - startWall                    // Wall-clock time
         print("CPU time (clock):", cpuTime)
         print("Wall-clock time (MPI_Wtime):", wallTime)
    else if rank == 1:
         for count from 0 to PING_PONG_LIMIT - 1:
              MPI_Recv(message, source=0)
              MPI_Send(message, destination=0)
    MPI_Finalize()

// Note:
// clock() reports a nonzero value after approximately 1/CLOCKS_PER_SEC seconds (typically about 1 microsecond).
// clock() measures CPU time while MPI_Wtime() measures the full elapsed (wall-clock) time.
```

────────────────────────────
**Question 9: Measuring Redistribution Cost Between Block and Cyclic Distributions**

*Question:*  
Imagine a large vector is distributed among processes. Write pseudocode to measure the time cost for redistributing the vector between block and cyclic distributions. Also, discuss factors influencing the redistribution time.

*Answer (Pseudocode):*

```
// Assume global vector V of size N distributed among P processes
function measureRedistributionCost():
    // --- Block-to-Cyclic Redistribution ---
    startTime = MPI_Wtime()
    for each local element with global index i in block:
         targetProcess = i mod P
         if (targetProcess != myRank):
              MPI_Send(element, targetProcess)
         else:
              cyclicBuffer.add(element)
    while (not all expected elements received):
         element = MPI_Recv(anySource)
         cyclicBuffer.add(element)
    blockToCyclicTime = MPI_Wtime() - startTime
    print("Block-to-Cyclic Redistribution Time:", blockToCyclicTime)

    // --- Cyclic-to-Block Redistribution ---
    startTime = MPI_Wtime()
    for each element in cyclicBuffer with global index i:
         targetProcess = floor(i / (N / P))
         if (targetProcess != myRank):
              MPI_Send(element, targetProcess)
         else:
              blockBuffer.add(element)
    while (not all expected elements received):
         element = MPI_Recv(anySource)
         blockBuffer.add(element)
    cyclicToBlockTime = MPI_Wtime() - startTime
    print("Cyclic-to-Block Redistribution Time:", cyclicToBlockTime)
    
// Factors influencing time:
// - Size of vector (N)
// - Number of processes (P)
// - Message latency and network bandwidth
// - Communication overhead and efficiency of MPI routines
```

────────────────────────────
**Question 10: Thread-Safe Tokenizer Without Modifying the Original String**

*Question:*  
Design a thread-safe tokenizer that extracts tokens from a string without modifying the original input. Provide pseudocode and outline your implementation considerations.

*Answer (Pseudocode):*

```
// Thread-safe tokenizer function
function threadSafeTokenizer(input, delimiters, savePtr):
    if input is not NULL:
         savePtr = input
    if savePtr is NULL or at end of string:
         return NULL
    start = savePtr
    pos = findFirstOccurrence(savePtr, delimiters)  // Similar to strpbrk
    if pos is NULL:
         token = duplicateString(start)
         savePtr = NULL
    else:
         tokenLength = pos - start
         token = allocateMemory(tokenLength + 1)
         copy substring of length tokenLength from start into token
         token[tokenLength] = '\0'
         savePtr = pos + 1
    return token

// Usage in a thread-safe context:
function processString(input, delimiters):
    savePtr = NULL
    token = threadSafeTokenizer(input, delimiters, savePtr)
    while token is not NULL:
         process(token)  // e.g., print or analyze token
         free(token)
         token = threadSafeTokenizer(NULL, delimiters, savePtr)
```

*Implementation Considerations:*  
- **Thread Safety:** No shared mutable state is used; each thread maintains its own `savePtr`.  
- **Immutability:** The original input string is preserved.  
- **Memory Management:** New memory is allocated for each token, so the caller must free it.

────────────────────────────
**Question 11: MPI Block-Column Distribution for Matrix-Vector Multiplication**

*Question:*  
Describe how you would implement block-column distribution and matrix-vector multiplication in MPI. Include steps for initializing MPI, distributing data, performing local multiplication, reducing results, and finalizing.

*Answer (Pseudocode):*

```
// Main pseudocode
function distributedMatrixVectorMultiply():
    MPI_Init()
    rank = MPI_Comm_rank()
    size = MPI_Comm_size()
    if (rank == 0):
         n = readMatrixOrder()         // Order of the matrix
         A = readFullMatrix()            // n x n matrix
         x = readVector()                // Vector of size n
         // For each process, compute localCols and offset based on block-column distribution
         for i from 0 to size - 1:
              localCols = computeLocalColumns(n, i, size)
              if i == 0:
                   localA = extractBlockColumns(A, start=0, count=localCols)
              else:
                   temp = packBlockColumns(A, start=offset, count=localCols)
                   MPI_Send(temp, destination=i)
         MPI_Bcast(x, root=0)
    else:
         localCols = computeLocalColumns(n, rank, size)
         localA = allocateMatrix(n, localCols)
         MPI_Recv(localA, source=0)
         MPI_Bcast(x, root=0)
    // Each process extracts its corresponding subvector for its block columns
    offset = computeOffset(rank, n, size)
    x_local = extractSubVector(x, offset, localCols)
    // Local matrix-vector multiplication: y_local = localA * x_local
    y_local = localMatrixVectorMultiply(localA, x_local)
    // Global reduction to sum partial results
    global_y = MPI_Reduce(y_local, operation=SUM, root=0)
    if (rank == 0):
         print("Resulting vector y:", global_y)
    MPI_Finalize()
```

────────────────────────────
**Question 12: OpenMP Donation Collection Simulation**

*Question:*  
A community group with 10 volunteers collects donations from 10 city regions. Each volunteer collects a random donation amount. Design an OpenMP program that simulates this process and aggregates the total donation.

*Answer (Pseudocode):*

```
// Global array to store donations
NUM_VOLUNTEERS = 10
donations[NUM_VOLUNTEERS] = {0}

// Set the number of threads equal to NUM_VOLUNTEERS
parallel region:
    threadId = omp_get_thread_num()
    seed = currentTime() + threadId
    donation = generateRandomDonation(seed)  // e.g., random value between 0 and 1000.99
    donations[threadId] = donation
    print("Volunteer", threadId, "collected:", donation)
// After parallel region, aggregate the total donation
total = 0
for i from 0 to NUM_VOLUNTEERS - 1:
    total += donations[i]
print("Total funds raised:", total)
```

────────────────────────────
**Question 13: Aggregating Local Temperature Extremes with MPI**

*Question:*  
Dr. Estelle collects temperature data processed by regional teams. Each team computes local minimum and maximum temperatures. Using MPI, how do you aggregate these into global minimum and maximum values?

*Answer (Pseudocode):*

```
// In each process:
function aggregateTemperatureExtremes():
    localMin = computeLocalMinTemperature()
    localMax = computeLocalMaxTemperature()
    globalMin = MPI_Reduce(localMin, operation=MIN, root=0)
    globalMax = MPI_Reduce(localMax, operation=MAX, root=0)
    if (rank == 0):
         print("Global Temperature Range: Min =", globalMin, "Max =", globalMax)
    MPI_Finalize()
```

────────────────────────────
**Question 14: QuickBuy Simulation Using Pthreads**

*Question:*  
QuickBuy, an online shopping platform, allows only 10 customers to buy a discounted product. Using multithreading with Pthreads, simulate a scenario where more than 10 customer threads attempt to purchase the product, but only 10 succeed. Compare busy waiting versus mutex synchronization.

*Answer (Pseudocode):*

```
// Version A: Busy Waiting (No Mutex)
Global available = 10
function customerThread_Busy():
    id = threadID
    while true:
         // No locking: check available offers
         if available > 0:
              // Race condition may occur without synchronization
              available = available - 1
              print("Customer", id, "succeeded!")
              break
         // Busy-wait: continuously check the shared variable

// Version B: Using Mutex Synchronization
Global available = 10
mutex = new Mutex()
function customerThread_Mutex():
    id = threadID
    lock(mutex)
    if available > 0:
         available = available - 1
         print("Customer", id, "succeeded!")
    else:
         print("Customer", id, "failed to buy.")
    unlock(mutex)

// Main function for both versions:
function main():
    for id from 1 to 20:
         createThread(customerThread)  // Choose Busy or Mutex version
    wait for all threads to finish
```

*Explanation:*  
- **Busy Waiting:** Threads continuously check a shared variable without synchronization, which may lead to race conditions and allow more than 10 successes.  
- **Mutex Synchronization:** A mutex ensures that only one thread at a time checks and updates the available count, ensuring exactly 10 customers succeed.

────────────────────────────
This complete text includes all questions and pseudocode answers. You can now generate your PDF using your own tool or script.
