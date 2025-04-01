Below is a revised version of the answers. Each section includes the complete question details along with a brief theoretical explanation for theory parts and concise pseudocode outlines for parts that require code-like solutions.

────────────────────────────
**Question 1: Approaches to Writing Parallel Programs and MIMD Systems**

*Question:*  
In modern computing, performance improvements rely heavily on parallel programming techniques. Describe different approaches to writing parallel programs, providing examples of each. Additionally, explain how parallel systems are classified and differentiate them from concurrent and distributed systems. How do the principal types of MIMD systems contribute to efficient parallel computing?

*Answer (Theoretical):*  
Parallel programming can be achieved using several approaches:  
- **Thread-Based (Shared Memory):** Multiple threads operate concurrently within a single address space (e.g., using OpenMP or Pthreads).  
- **Message Passing (Distributed Memory):** Independent processes communicate by sending messages (e.g., MPI).  
- **Data Parallelism:** The same operation is applied concurrently across elements (e.g., vectorized operations or GPU programming).  
- **Task Parallelism:** Different tasks run concurrently.  

Parallel systems are classified as shared memory, distributed memory, or hybrid systems. In MIMD systems, each processor can execute different instructions on different data, which allows scalable and efficient parallel computing.

*Pseudocode Outline:*

```
// Thread-Based Parallelism Example (Shared Memory)
function threadParallelExample():
    for each thread in THREAD_POOL:
         parallel_execute:
              print("Hello from thread", thread.id)

// Message Passing Example (Distributed Memory)
function mpiExample():
    MPI_Init()
    rank = MPI_Comm_rank()
    if rank == 0:
         print("Hello from process", rank)
    else:
         performLocalWork()
    MPI_Finalize()
```

────────────────────────────
**Question 2: Von Neumann Architecture and Memory Models**

*Question:*  
Explain the Von Neumann architecture and its significance in modern computing. How does it relate to parallel and distributed systems? Additionally, compare shared memory and distributed memory MIMD systems.

*Answer:*  
The Von Neumann architecture is based on the stored-program concept where both instructions and data share a common memory. Its simplicity led to early computer designs but also introduces a bottleneck (the "Von Neumann bottleneck") due to a single data path. Modern processors overcome these limitations by incorporating multiple cores and parallelism. In shared memory MIMD systems, multiple cores access common memory, while in distributed memory systems, each processor has its own memory and communicates via message passing.

*Essential Outline (Pseudocode):*

```
// Von Neumann Execution Cycle
function vonNeumannCycle():
    load_program_and_data_from_memory()
    while not end_of_program:
         instruction = fetch_from_memory()
         decoded = decode(instruction)
         execute(decoded)
```

────────────────────────────
**Question 3: Cloud-Based AI Training Platform**

*Question:*  
A cloud-based AI training platform is designed to process large-scale machine learning workloads by distributing computations across multiple independent servers, each with its own private memory. The system must efficiently exchange model parameters between nodes while minimizing communication overhead. In this scenario, which type of MIMD system would be ideal, and how does it handle data exchange between processors?

*Answer (Theoretical):*  
A distributed memory MIMD system is ideal. Each server computes local model updates, and efficient data exchange is handled using message-passing protocols (such as MPI_Allreduce) to aggregate and synchronize model parameters across nodes.

*Pseudocode Outline:*

```
// Distributed AI Training using MPI
function distributedAITraining():
    MPI_Init()
    local_data = getLocalData()
    local_update = computeLocalModelUpdate(local_data)
    global_update = MPI_Allreduce(local_update, operation=SUM)
    updateModel(global_update)
    MPI_Finalize()
```

────────────────────────────
**Question 4: Pthreads Dynamic Task Queue**

*Question:*  
You are tasked with developing a multi-threaded system using Pthreads to manage a dynamic task queue. The program should start with a user-specified number of worker threads that initially sleep in a condition wait. The main thread generates blocks of tasks and wakes a worker thread with a condition signal. After completing its tasks, the worker returns to waiting. Once all tasks are generated, the main thread signals completion and uses a condition broadcast to wake all threads for termination. Implement the system with proper synchronization.

*Answer (Theoretical):*  
A dynamic task queue can be implemented using a mutex to protect shared access and a condition variable to signal worker threads when tasks become available. A global flag indicates when no more tasks will be generated so workers can exit cleanly.

*Pseudocode Outline:*

```
// Global shared variables
taskQueue = empty queue
mutex = new Mutex()
conditionVar = new ConditionVariable()
tasksDone = false

function enqueueTask(task):
    lock(mutex)
    taskQueue.enqueue(task)
    signal(conditionVar)
    unlock(mutex)

function workerThread(threadId):
    while true:
         lock(mutex)
         while taskQueue.isEmpty() and not tasksDone:
              wait(conditionVar, mutex)
         if taskQueue.isEmpty() and tasksDone:
              unlock(mutex)
              break
         task = taskQueue.dequeue()
         unlock(mutex)
         process(task)
         sleep(1)
    print("Thread", threadId, "exiting")
```

────────────────────────────
**Question 5: Handling Imbalanced Workload in Matrix-Vector Multiplication**

*Question:*  
Imagine you are implementing a parallel matrix-vector multiplication algorithm for a large dataset. Initially, the matrix dimensions are perfectly divisible by the number of available threads. However, if the dimensions do not divide evenly, describe an approach to handle such an imbalanced workload efficiently.

*Answer (Theoretical):*  
Compute the base number of rows each thread should process using integer division, and then distribute the remaining extra rows among the first few threads. This ensures a balanced workload with minimal idle time.

*Pseudocode Outline:*

```
// m: total rows, t: total threads
function assignRows(m, t):
    base = m / t              // Integer division
    remainder = m mod t
    for i from 0 to t - 1:
         start = i * base + min(i, remainder)
         count = base + (if i < remainder then 1 else 0)
         end = start + count - 1
         assignTaskToThread(i, start, end)
```

────────────────────────────
**Question 6: OpenMP Histogram Program**

*Question:*  
Develop an OpenMP program to implement a histogram where the master thread receives sensor data in the range 0–99, and slave threads (each representing a bucket) count the data falling within their range. The master thread displays bucket counts periodically.

*Answer (Theoretical):*  
The master thread generates sensor data and updates a shared array, while each worker thread (bucket) uses atomic operations to update its count for its specific data range. Periodic printing shows the current state of the histogram.

*Pseudocode Outline:*

```
// Global: sensorData, bucketCounts[NUM_BUCKETS], dataIndex
parallel region with NUM_BUCKETS+1 threads:
    threadId = omp_get_thread_num()
    if threadId == 0: // Master
         for i from 0 to TOTAL_DATA-1:
              sensorData[i] = randomInteger(0, 99)
              dataIndex++
              if (i mod 100 == 0): print(bucketCounts)
              sleep(delay)
    else: // Worker for bucket (threadId-1)
         bucket = threadId - 1
         lowerBound = bucket * (100 / NUM_BUCKETS)
         upperBound = lowerBound + (100 / NUM_BUCKETS)
         myIndex = 0
         while true:
              localIndex = dataIndex
              while myIndex < localIndex:
                    if sensorData[myIndex] in [lowerBound, upperBound):
                         atomic_increment(bucketCounts[bucket])
                    myIndex++
              if localIndex >= TOTAL_DATA: break
              sleep(shortDelay)
```

────────────────────────────
**Question 7: MPI Global Min/Max Shipping Times**

*Question:*  
You are running a large e-commerce operation where each warehouse computes the minimum and maximum shipping times. Using MPI, write a program flow to collect these local min/max values and compute the global minimum and maximum shipping times.

*Answer (Theoretical):*  
Each process computes local shipping time extremes. Then, MPI_Reduce is used with the MPI_MIN operator for the global minimum and MPI_MAX for the global maximum, aggregating the data at the root process.

*Pseudocode Outline:*

```
// In each process:
function aggregateShippingTimes():
    MPI_Init()
    localMin = computeLocalMinShippingTime()
    localMax = computeLocalMaxShippingTime()
    globalMin = MPI_Reduce(localMin, operation=MIN, root=0)
    globalMax = MPI_Reduce(localMax, operation=MAX, root=0)
    if rank == 0:
         print("Global Min:", globalMin, "Global Max:", globalMax)
    MPI_Finalize()
```

────────────────────────────
**Question 8: Ping-Pong Communication and Timing**

*Question:*  
You implement a ping-pong communication pattern between two processes to measure message passing cost. How long does the program need to run for clock() to report a nonzero runtime? How do timings from clock() compare to those from MPI_Wtime()?

*Answer (Theoretical):*  
The clock() function measures CPU time and will report a nonzero value after roughly 1/CLOCKS_PER_SEC seconds (usually about 1 microsecond). MPI_Wtime() measures elapsed wall-clock time, which includes waiting periods and provides a more accurate measure for communication tests.

*Pseudocode Outline:*

```
// Process 0:
function pingPongTest():
    MPI_Init()
    if rank == 0:
         startWall = MPI_Wtime()
         startCPU = clock()
         for count from 0 to LIMIT-1:
              MPI_Send(message, destination=1)
              MPI_Recv(message, source=1)
         endCPU = clock()
         endWall = MPI_Wtime()
         print("CPU time:", (endCPU - startCPU) / CLOCKS_PER_SEC)
         print("Wall time:", endWall - startWall)
    else:
         for count from 0 to LIMIT-1:
              MPI_Recv(message, source=0)
              MPI_Send(message, destination=0)
    MPI_Finalize()
```

────────────────────────────
**Question 9: Measuring Redistribution Cost Between Block and Cyclic Distributions**

*Question:*  
Imagine a large vector is distributed among processes. Write pseudocode to measure the time cost for redistributing the vector between block and cyclic distributions. Also, discuss factors influencing the redistribution time.

*Answer (Theoretical):*  
Measure redistribution time by capturing the wall-clock time before and after the redistribution. The time cost is influenced by the vector size, the number of processes, network latency, bandwidth, and overhead in message handling.

*Pseudocode Outline:*

```
// For each redistribution:
function measureRedistributionCost():
    startTime = MPI_Wtime()
    for each local element with global index i in block:
         targetProcess = i mod P
         if targetProcess != myRank:
              MPI_Send(element, targetProcess)
         else:
              cyclicBuffer.add(element)
    while not all expected elements received:
         element = MPI_Recv(anySource)
         cyclicBuffer.add(element)
    blockToCyclicTime = MPI_Wtime() - startTime
    print("Block-to-Cyclic Time:", blockToCyclicTime)
    
    // Similarly, for cyclic-to-block redistribution.
```

────────────────────────────
**Question 10: Thread-Safe Tokenizer Without Modifying the Original String**

*Question:*  
Design a thread-safe tokenizer that extracts tokens from a string without modifying the original input. Provide pseudocode and outline your implementation considerations.

*Answer (Theoretical):*  
The tokenizer should maintain its own state (using a pointer) and return newly allocated tokens so that the original string remains unchanged. Each thread uses its own state variable to ensure thread safety. Memory management is crucial, as each token must be freed after use.

*Pseudocode Outline:*

```
// Thread-safe tokenizer function
function threadSafeTokenizer(input, delimiters, savePtr):
    if input is not NULL:
         savePtr = input
    if savePtr is NULL or at end of string:
         return NULL
    start = savePtr
    pos = findFirstOccurrence(savePtr, delimiters)
    if pos is NULL:
         token = duplicateString(start)
         savePtr = NULL
    else:
         tokenLength = pos - start
         token = allocateMemory(tokenLength + 1)
         copy substring into token
         token[tokenLength] = '\0'
         savePtr = pos + 1
    return token

// Usage:
function processString(input, delimiters):
    savePtr = NULL
    token = threadSafeTokenizer(input, delimiters, savePtr)
    while token is not NULL:
         process(token)
         free(token)
         token = threadSafeTokenizer(NULL, delimiters, savePtr)
```

────────────────────────────
**Question 11: MPI Block-Column Distribution for Matrix-Vector Multiplication**

*Question:*  
Describe how you would implement block-column distribution and matrix-vector multiplication in MPI. Include steps for initializing MPI, distributing data, performing local multiplication, reducing results, and finalizing.

*Answer (Theoretical):*  
Process 0 reads the entire matrix and vector, partitions the matrix into block columns, and sends each block to the appropriate process. Each process multiplies its submatrix by the corresponding portion of the vector. Finally, MPI_Reduce is used to sum the partial results to form the final output vector.

*Pseudocode Outline:*

```
// Main pseudocode
function distributedMatrixVectorMultiply():
    MPI_Init()
    if rank == 0:
         n = readMatrixOrder()
         A = readFullMatrix()      // n x n matrix
         x = readVector()          // Vector of size n
         for each process i:
              localCols = computeLocalColumns(n, i, size)
              if i == 0:
                   localA = extractBlockColumns(A, start=0, count=localCols)
              else:
                   temp = packBlockColumns(A, start=offset, count=localCols)
                   MPI_Send(temp, destination=i)
         MPI_Bcast(x, root=0)
    else:
         localA = MPI_Recv(from=0)
         MPI_Bcast(x, root=0)
    x_local = extractSubVector(x, offset, localCols)
    y_local = localMatrixVectorMultiply(localA, x_local)
    global_y = MPI_Reduce(y_local, operation=SUM, root=0)
    if rank == 0:
         print("Resulting vector y:", global_y)
    MPI_Finalize()
```

────────────────────────────
**Question 12: OpenMP Donation Collection Simulation**

*Question:*  
A community group with 10 volunteers collects donations from 10 city regions. Each volunteer collects a random donation amount. Design an OpenMP program that simulates this process and aggregates the total donation.

*Answer (Theoretical):*  
Each volunteer is simulated by a thread that collects a random donation amount. The individual donations are stored in an array and then summed to compute the total funds raised.

*Pseudocode Outline:*

```
// Global array for donations
donations[10] = {0}

parallel region with 10 threads:
    threadId = omp_get_thread_num()
    seed = currentTime() + threadId
    donation = generateRandomDonation(seed)  // e.g., value between $0.00 and $1000.99
    donations[threadId] = donation
    print("Volunteer", threadId, "collected:", donation)

total = sum(donations)
print("Total funds raised:", total)
```

────────────────────────────
**Question 13: Aggregating Local Temperature Extremes with MPI**

*Question:*  
Dr. Estelle collects temperature data processed by regional teams. Each team computes local minimum and maximum temperatures. Using MPI, how do you aggregate these into global minimum and maximum values?

*Answer (Theoretical):*  
Each process computes its local temperature extremes. MPI_Reduce is then used with the MPI_MIN operator to compute the global minimum and with MPI_MAX to compute the global maximum. The root process collects and displays the results.

*Pseudocode Outline:*

```
// In each process:
function aggregateTemperatureExtremes():
    localMin = computeLocalMinTemperature()
    localMax = computeLocalMaxTemperature()
    globalMin = MPI_Reduce(localMin, operation=MIN, root=0)
    globalMax = MPI_Reduce(localMax, operation=MAX, root=0)
    if rank == 0:
         print("Global Temperature Range: Min =", globalMin, "Max =", globalMax)
    MPI_Finalize()
```

────────────────────────────
**Question 14: QuickBuy Simulation Using Pthreads**

*Question:*  
QuickBuy, an online shopping platform, allows only 10 customers to buy a discounted product. Using multithreading with Pthreads, simulate a scenario where more than 10 customer threads attempt to purchase the product, but only 10 succeed. Compare busy waiting versus mutex synchronization.

*Answer (Theoretical):*  
In a busy-wait version, threads repeatedly check a shared counter without synchronization, which may lead to race conditions. With mutex synchronization, a mutex ensures that only one thread decrements the counter at a time, guaranteeing that exactly 10 customers succeed.

*Pseudocode Outline:*

```
// Version A: Busy Waiting (No Mutex)
Global available = 10
function customerThread_Busy():
    id = threadID
    while true:
         if available > 0:
              available = available - 1  // May cause race conditions
              print("Customer", id, "succeeded!")
              break

// Version B: Using Mutex Synchronization
Global available = 10, mutex = new Mutex()
function customerThread_Mutex():
    id = threadID
    lock(mutex)
    if available > 0:
         available = available - 1
         print("Customer", id, "succeeded!")
    else:
         print("Customer", id, "failed to buy.")
    unlock(mutex)

// Main:
function main():
    for id from 1 to 20:
         createThread(customerThread)  // Choose either the busy or mutex version
    wait for all threads to finish
```

────────────────────────────
This complete text includes the full details of each question along with brief theoretical explanations and concise pseudocode outlines where needed. You can now use this text to generate your PDF or for further study.
