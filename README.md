Below is a complete, structured document covering all 17 questions from the assignment. For each question, the full text is provided, followed by a brief theoretical explanation (for those that are concept‐based) and descriptive pseudocode with detailed comments so that a layperson can follow the logic. References to the source file are included as inline citations (for example, citeturn2file9 for Question 15).

────────────────────────────
**Question 1: Approaches to Writing Parallel Programs and MIMD Systems**

*Question:*  
In modern computing, performance improvements rely heavily on parallel programming techniques. Describe different approaches to writing parallel programs, providing examples of each. Additionally, explain how parallel systems are classified and differentiate them from concurrent and distributed systems. How do the principal types of MIMD systems contribute to efficient parallel computing?

*Answer (Theoretical):*  
Parallel programming can be implemented using several approaches:  
- **Thread-Based Parallelism (Shared Memory):** Multiple threads run concurrently within a single address space (using frameworks like OpenMP or Pthreads).  
- **Message Passing (Distributed Memory):** Independent processes communicate explicitly by sending messages (using MPI, for example).  
- **Data Parallelism:** The same operation is applied simultaneously to many data elements (often used in vectorized computations and GPU programming).  
- **Task Parallelism:** Different tasks or functions are executed concurrently.  

MIMD (Multiple Instruction, Multiple Data) systems allow each processor to execute different instructions on different pieces of data. They are implemented in shared memory, distributed memory, or hybrid forms, each contributing to scalability and efficient workload distribution.

*Descriptive Pseudocode:*

```
// Thread-Based Parallelism Example
function threadParallelExample():
    // For each thread in a pre-created thread pool:
    for each thread in THREAD_POOL:
         // Run this code concurrently:
         parallel_execute:
              print("Hello from thread", thread.id)

// Message Passing Example using MPI
function mpiExample():
    MPI_Init()                              // Start MPI environment
    rank = MPI_Comm_rank()                  // Get process identifier
    if rank == 0:
         print("Hello from process", rank)  // Master process prints a message
    else:
         performLocalWork()                  // Other processes execute their work
    MPI_Finalize()                          // End MPI environment
```

────────────────────────────
**Question 2: Von Neumann Architecture and Memory Models**

*Question:*  
Computer architecture plays a crucial role in designing efficient computing systems. Explain the Von Neumann architecture and its significance in modern computing. How does it relate to parallel and distributed systems? Additionally, compare shared memory and distributed memory MIMD systems.

*Answer:*  
The Von Neumann architecture features a single memory space that holds both data and program instructions. This design simplifies computer construction but creates a bottleneck since the CPU fetches instructions and data over the same bus. Modern computers extend this model by using multiple cores, pipelining, and caches. In shared memory systems, all cores access a common memory area; in distributed memory systems, each processor has its own local memory and communicates via message passing.

*Descriptive Pseudocode:*

```
// Basic Von Neumann Execution Cycle
function vonNeumannCycle():
    load_program_and_data_from_memory()  // Load code and data
    while not end_of_program:
         instruction = fetch_from_memory()  // Get next instruction
         decoded = decode(instruction)      // Decode instruction
         execute(decoded)                   // Execute the instruction
```

────────────────────────────
**Question 3: Cloud-Based AI Training Platform**

*Question:*  
A cloud-based AI training platform is designed to process large-scale machine learning workloads by distributing computations across multiple independent servers, each with its own private memory. The system must efficiently exchange model parameters between nodes while minimizing communication overhead. In this scenario, which type of MIMD system would be ideal, and how does it handle data exchange between processors?

*Answer:*  
A distributed memory MIMD system is ideal because each server works independently using its private memory. Data exchange is handled using message-passing libraries (like MPI) that employ efficient collective operations (for example, MPI_Allreduce) to aggregate model updates with minimal overhead.

*Descriptive Pseudocode:*

```
// Distributed AI Training using MPI
function distributedAITraining():
    MPI_Init()                                // Initialize MPI
    local_data = getLocalData()               // Read local machine learning data
    local_update = computeLocalModelUpdate(local_data)  // Compute model update locally
    // All nodes combine their updates using a collective sum operation
    global_update = MPI_Allreduce(local_update, operation=SUM)
    updateModel(global_update)                // Update the global model with the aggregated update
    MPI_Finalize()                            // Finalize MPI
```

*(Reference: citeturn2file0)*

────────────────────────────
**Question 4: Pthreads Dynamic Task Queue**

*Question:*  
You are tasked with developing a multi-threaded system using Pthreads to manage a dynamic task queue. The program should start with a user-specified number of worker threads that initially sleep in a condition wait. The main thread generates blocks of tasks (using linked list operations) and wakes a worker thread with a condition signal. After completing its tasks, the worker returns to the waiting state. Once all tasks are generated, the main thread updates a global variable to signal completion and uses a condition broadcast to wake all threads for termination. Implement the system with proper synchronization, ensuring efficient task distribution and clean thread shutdown.

*Answer:*  
This solution uses a linked list as the task queue. A mutex is used to protect the shared queue, and a condition variable allows worker threads to sleep until new tasks are available. A global flag indicates when all tasks have been generated so that threads can terminate gracefully.

*Descriptive Pseudocode:*

```
// Global variables:
taskQueue = empty queue           // Linked list for tasks
mutex = new Mutex()               // Mutex to protect queue access
conditionVar = new ConditionVariable()  // Condition variable for signaling
tasksDone = false                 // Flag indicating task generation is complete

// Function to enqueue a new task:
function enqueueTask(task):
    lock(mutex)                   // Lock the queue
    taskQueue.enqueue(task)       // Add task to the queue
    signal(conditionVar)          // Wake one waiting worker
    unlock(mutex)                 // Unlock the queue

// Worker thread function:
function workerThread(threadId):
    while true:
         lock(mutex)
         // Wait until a task is available or all tasks are done:
         while taskQueue.isEmpty() and not tasksDone:
              wait(conditionVar, mutex)
         if taskQueue.isEmpty() and tasksDone:
              unlock(mutex)
              break              // Exit if no tasks remain
         task = taskQueue.dequeue()  // Remove a task from the queue
         unlock(mutex)
         process(task)         // Process the task (simulate with sleep)
         sleep(1)              // Simulate work duration
    print("Thread", threadId, "exiting")
```

────────────────────────────
**Question 5: Handling Imbalanced Workload in Matrix-Vector Multiplication**

*Question:*  
Imagine you are implementing a parallel matrix-vector multiplication algorithm for a large dataset. In your initial design, you assumed that the matrix dimensions—the number of rows (m) and columns (n)—are perfectly divisible by the number of available threads (t). However, in a real-world scenario, the matrix dimensions might not divide evenly among the threads. For example, if you have a matrix with 10 rows and 3 threads, not all threads would get the same number of rows to process. Describe an approach to handle such an imbalanced workload efficiently.

*Answer:*  
You can use static partitioning with remainder adjustment. First, calculate the base number of rows per thread using integer division. Then, assign one extra row to the first “remainder” threads so that the workload is as balanced as possible. Alternatively, you can use a dynamic scheduling approach with a shared work queue.

*Descriptive Pseudocode:*

```
// m: total rows; t: total threads
function assignRows(m, t):
    base = m / t              // Basic number of rows per thread (integer division)
    remainder = m mod t       // Extra rows that cannot be evenly divided
    for i from 0 to t - 1:
         // Start index for thread i:
         start = i * base + min(i, remainder)
         // Each thread gets an extra row if its index is less than remainder
         if i < remainder:
             count = base + 1
         else:
             count = base
         end = start + count - 1  // Last row index assigned
         assignTaskToThread(i, start, end)  // Assign these rows to thread i
```

────────────────────────────
**Question 6: OpenMP Histogram Program**

*Question:*  
OpenMP Histogram Program: The master thread receives sensor data items in the range 0–99, and the data is made available to all slave threads. Each thread represents a bucket of values dynamically allocated based on the number of slave threads. If ten slave threads are started, the data items that fall in the range Bucket 0–9 are counted by Rank 0, and Bucket 90–99 is counted by Rank 9. The master thread should display each bucket count frequently. Implement this using OpenMP.

*Answer:*  
The master thread simulates incoming sensor data and writes it to a shared buffer. Each slave thread (bucket) checks the shared data for values within its range and atomically increments its count. The master thread periodically prints the histogram counts.

*Descriptive Pseudocode:*

```
// Global variables:
sensorData = array[TOTAL_DATA]       // Shared sensor data buffer
bucketCounts = array[NUM_BUCKETS] = {0}  // Histogram counts for each bucket
dataIndex = 0                        // Index of the next data item

parallel region with NUM_BUCKETS + 1 threads:
    threadId = omp_get_thread_num()
    if threadId == 0:
         // Master thread: generates sensor data.
         for i from 0 to TOTAL_DATA - 1:
              sensorData[i] = randomInteger(0, 99)
              dataIndex++          // Update count of produced items
              if (i mod 100 == 0):
                  print(bucketCounts)  // Display histogram periodically
              sleep(delay)         // Simulate delay between data items
    else:
         // Slave threads: each represents one bucket.
         bucket = threadId - 1
         lowerBound = bucket * (100 / NUM_BUCKETS)
         upperBound = lowerBound + (100 / NUM_BUCKETS)
         myIndex = 0             // Tracks how many data items have been processed
         while true:
              localIndex = dataIndex
              while myIndex < localIndex:
                    if sensorData[myIndex] >= lowerBound and sensorData[myIndex] < upperBound:
                         atomic_increment(bucketCounts[bucket])
                    myIndex++   // Process next data item
              if localIndex >= TOTAL_DATA:
                    break     // Exit when all data are processed
              sleep(shortDelay)  // Reduce busy waiting
```

*(Reference: citeturn2file15)*

────────────────────────────
**Question 7: MPI Global Min/Max Shipping Times**

*Question:*  
Imagine you are running a large e-commerce operation where shipping times are recorded daily at multiple warehouse hubs worldwide. Each local warehouse manager computes the minimum and maximum shipping times for their region. Your goal is to determine the global minimum and maximum shipping times across all warehouses to optimize your company’s delivery logistics. Using MPI for parallel processing, write a program flow with proper MPI system calls that collect these local minimum and maximum shipping times from each regional hub and compute the overall global minimum and maximum.

*Answer:*  
Each MPI process computes its local minimum and maximum shipping times from its data. Then, MPI_Reduce is used twice—once with the MPI_MIN operator and once with MPI_MAX—to compute the global minimum and maximum. The root process prints the final values.

*Descriptive Pseudocode:*

```
// Each process executes the following:
function aggregateShippingTimes():
    MPI_Init()                                  // Initialize MPI environment
    localMin = computeLocalMinShippingTime()    // Compute local minimum shipping time
    localMax = computeLocalMaxShippingTime()    // Compute local maximum shipping time
    // Reduce to obtain global minimum:
    globalMin = MPI_Reduce(localMin, operation=MIN, root=0)
    // Reduce to obtain global maximum:
    globalMax = MPI_Reduce(localMax, operation=MAX, root=0)
    if rank == 0:
         print("Global Minimum Shipping Time:", globalMin)
         print("Global Maximum Shipping Time:", globalMax)
    MPI_Finalize()                              // Finalize MPI environment
```

*(Reference: citeturn2file0)*

────────────────────────────
**Question 8: Ping-Pong Communication and Timing**

*Question:*  
You are testing the communication performance of a distributed system. In your experiment, you implement a "ping-pong" communication pattern: Process A sends a message to process B ("ping"), and process B responds ("pong"). To estimate the message-passing cost, you decide to time repeated ping-pong exchanges using the C clock() function. Answer the following:  
• How long does the program need to run for clock() to report a nonzero runtime?  
• How do the timings from clock() compare to those from MPI_Wtime()?

*Answer:*  
The clock() function reports CPU time and typically registers a nonzero value after about 1/CLOCKS_PER_SEC seconds (roughly 1 microsecond on many systems). However, clock() only measures active CPU execution time, whereas MPI_Wtime() measures the elapsed (wall-clock) time, including waiting periods. Thus, MPI_Wtime() provides a more accurate measure of communication delays.

*Descriptive Pseudocode:*

```
// In process 0 (the initiator):
function pingPongTest():
    MPI_Init()                              // Start MPI environment
    if rank == 0:
         startWall = MPI_Wtime()            // Record wall-clock start time
         startCPU = clock()                 // Record CPU start time
         for count from 0 to LIMIT - 1:
              MPI_Send(message, destination=1)  // Send message to process 1
              MPI_Recv(message, source=1)         // Receive reply from process 1
         endCPU = clock()                   // Record CPU end time
         endWall = MPI_Wtime()              // Record wall-clock end time
         cpuTime = (endCPU - startCPU) / CLOCKS_PER_SEC
         wallTime = endWall - startWall
         print("CPU time (clock):", cpuTime)
         print("Wall-clock time (MPI_Wtime):", wallTime)
    else:
         // Process 1 simply echoes messages back to process 0.
         for count from 0 to LIMIT - 1:
              MPI_Recv(message, source=0)
              MPI_Send(message, destination=0)
    MPI_Finalize()                          // End MPI environment
```

*(Reference: citeturn2file4)*

────────────────────────────
**Question 9: Measuring Redistribution Cost Between Block and Cyclic Distributions**

*Question:*  
Imagine you are working with a distributed system where a large vector is split across multiple processes. Initially, the vector elements are divided using a block distribution, where each process handles a contiguous chunk of elements. Now, your team wants to switch to a cyclic distribution, where elements are distributed in a round-robin fashion across processes—and possibly back to block distribution later.  
• Write pseudocode to measure the cost (in terms of time) for redistributing the vector between block and cyclic distributions.  
• How long does the redistribution take in each direction, and what factors influence this time?

*Answer:*  
The redistribution cost is measured by recording the wall-clock time before and after the data transfers. The time depends on factors such as vector size, number of processes, message latency, network bandwidth, and the overhead of packing/unpacking noncontiguous data.

*Descriptive Pseudocode:*

```
// Function to measure redistribution cost:
function measureRedistributionCost():
    // --- Block-to-Cyclic Redistribution ---
    startTime = MPI_Wtime()  // Record start time
    for each local element with global index i in the block:
         targetProcess = i mod P   // Determine target process based on cyclic distribution
         if targetProcess != myRank:
              MPI_Send(element, targetProcess)  // Send element to the appropriate process
         else:
              cyclicBuffer.add(element)         // Retain locally if assigned to self
    // Wait until all expected elements have been received:
    while not all expected elements received:
         element = MPI_Recv(anySource)
         cyclicBuffer.add(element)
    blockToCyclicTime = MPI_Wtime() - startTime
    print("Block-to-Cyclic Redistribution Time:", blockToCyclicTime)
    
    // --- Cyclic-to-Block Redistribution ---
    startTime = MPI_Wtime()  // Record start time for reverse redistribution
    for each element in cyclicBuffer with global index i:
         targetProcess = floor(i / (N / P))  // Calculate target process for block distribution
         if targetProcess != myRank:
              MPI_Send(element, targetProcess)
         else:
              newBlockBuffer.add(element)
    while not all expected elements received:
         element = MPI_Recv(anySource)
         newBlockBuffer.add(element)
    cyclicToBlockTime = MPI_Wtime() - startTime
    print("Cyclic-to-Block Redistribution Time:", cyclicToBlockTime)
```

*(Reference: citeturn2file6)*

────────────────────────────
**Question 10: Thread-Safe Tokenizer Without Modifying the Original String**

*Question:*  
You are developing a multi-threaded application that processes text data. While exploring string tokenization, you come across strtok_r, which is thread-safe but modifies the input string by replacing delimiters with null characters. This behavior makes it unsuitable if you need to preserve the original string.  
• Design a thread-safe tokenizer that extracts tokens from a string without altering the original input.  
• How would you implement this, and what considerations would you keep in mind?

*Answer:*  
The tokenizer should be reentrant, using a pointer (saveptr) to track progress without modifying the original string. Each token is dynamically allocated so the original remains intact, and the caller is responsible for freeing memory. Considerations include ensuring thread safety (each thread must have its own saveptr), proper memory management, and handling potential performance overhead due to copying.

*Descriptive Pseudocode:*

```
// Thread-safe tokenizer function:
function threadSafeTokenizer(input, delimiters, savePtr):
    if input is not NULL:
         savePtr = input          // Initialize pointer on first call
    if savePtr is NULL or at end of string:
         return NULL              // No more tokens
    start = savePtr              // Mark beginning of token
    pos = findFirstOccurrence(savePtr, delimiters)  // Locate next delimiter
    if pos is NULL:
         token = duplicateString(start)   // Duplicate remaining string as token
         savePtr = NULL                     // End of string reached
    else:
         tokenLength = pos - start          // Length of token
         token = allocateMemory(tokenLength + 1)  // Allocate space for token and null terminator
         copy substring from start into token for tokenLength characters
         token[tokenLength] = '\0'            // Null-terminate token
         savePtr = pos + 1                    // Move pointer past delimiter
    return token

// Usage example:
function processString(input, delimiters):
    savePtr = NULL
    token = threadSafeTokenizer(input, delimiters, savePtr)
    while token is not NULL:
         print("Token:", token)   // Process token (e.g., print it)
         free(token)              // Free allocated memory
         token = threadSafeTokenizer(NULL, delimiters, savePtr)  // Get next token
```

*(Reference: citeturn2file16)*

────────────────────────────
**Question 11: MPI Block-Column Distribution for Matrix-Vector Multiplication**

*Question:*  
Imagine you are working at a data analytics firm where you need to repeatedly multiply a massive square matrix (of order n) by a single vector to update certain predictive models. Because the matrix is so large, you decide to distribute it among multiple parallel processes using MPI, employing a block-column distribution strategy. Specifically, Process 0 reads the entire matrix file and sends different blocks of columns to the other processes. Each process, in turn, multiplies its own block of columns by the vector. The final step is to combine all parallel products to form the resulting output vector, for which MPI_Reduce_scatter can prove especially valuable. Describe how you would implement this block-column distribution and matrix-vector multiplication in MPI (initializing MPI, distributing data, performing local multiplications, reducing the results, and finalizing). As part of this description, outline the necessary MPI function calls and illustrate the workflow with concise code.

*Answer:*  
Process 0 reads the full matrix and vector, calculates the number of columns each process will handle (with adjustments if n is not exactly divisible by the number of processes), and sends each block of columns using MPI_Send. All processes then perform local matrix–vector multiplication and the partial results are combined using an MPI reduction (or MPI_Reduce_scatter) to form the final output vector.

*Descriptive Pseudocode:*

```
// Main function for distributed matrix-vector multiplication:
function distributedMatrixVectorMultiply():
    MPI_Init()                              // Initialize MPI
    rank = MPI_Comm_rank()
    size = MPI_Comm_size()
    if rank == 0:
         n = readMatrixOrder()              // Read matrix order from file
         A = readFullMatrix()               // Read full n x n matrix
         x = readVector()                   // Read vector of size n
         // Determine column distribution among processes:
         base = n / size
         rem = n mod size
         col_start = 0
         for i from 0 to size - 1:
              localCols = base + (if i < rem then 1 else 0)
              if i == 0:
                   localA = extractBlockColumns(A, start=col_start, count=localCols)
              else:
                   temp = packBlockColumns(A, start=col_start, count=localCols)
                   MPI_Send(temp, n*localCols, MPI_DOUBLE, i, 0, MPI_COMM_WORLD)
                   free(temp)
              col_start = col_start + localCols
         MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD)  // Broadcast vector x
    else:
         // Other processes receive their block of columns:
         compute localCols and offset similar to process 0
         allocate localA for n rows and localCols columns
         MPI_Recv(localA, n*localCols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE)
         allocate x of size n and MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD)
    // Extract local portion of x:
    x_local = extractSubVector(x, offset, localCols)
    // Compute local matrix-vector product:
    y_local = localMatrixVectorMultiply(localA, x_local)
    // Combine local results to form global vector y:
    global_y = MPI_Reduce(y_local, operation=SUM, root=0)
    if rank == 0:
         print("Resulting vector y:", global_y)
    MPI_Finalize()                         // Finalize MPI
```

*(Reference: citeturn2file10)*

────────────────────────────
**Question 12: OpenMP Donation Collection Simulation**

*Question:*  
A community group is planning a fundraiser for a local charity. The group consists of 10 volunteers, and they decide to split the city into 10 regions, with each volunteer collecting donations from their assigned region. To stay organized, they want to track both the individual amounts collected by each volunteer and the total funds raised. How would you design an OpenMP program to simulate this donation collection process, where each volunteer collects a random amount, and the results are aggregated to calculate the total donation?

*Answer:*  
Use an OpenMP parallel region with 10 threads (one per volunteer). Each thread computes a random donation using a thread-safe random generator (rand_r) with a unique seed. The individual donation amounts are stored in an array, and after the parallel region, the total is computed by summing these values.

*Descriptive Pseudocode:*

```
// Global array for donations:
donations = array[NUM_VOLUNTEERS] initialized to 0

// Set the number of threads to 10:
omp_set_num_threads(NUM_VOLUNTEERS)

#pragma omp parallel
    threadId = omp_get_thread_num()                // Each thread gets a unique ID
    seed = currentTime() + threadId                 // Create a thread-local seed
    donation = (rand_r(&seed) % 1001) + ((rand_r(&seed) % 100) / 100.0)  // Generate donation between $0.00 and $1000.99
    donations[threadId] = donation                  // Store the donation
    print("Volunteer", threadId, "collected: $", donation)

After parallel region:
total = sum(donations)                              // Compute total funds raised
print("Total funds raised: $", total)
```

*(Reference: citeturn2file18)*

────────────────────────────
**Question 13: Aggregating Local Temperature Extremes with MPI**

*Question:*  
Dr. Estelle, a climate scientist, is analyzing global temperature trends. She collects temperature data from sensors across multiple continents, with each continent’s data processed by a different research team. Every team computes the minimum and maximum temperatures for their region. Now, Dr. Estelle needs to determine the global minimum and maximum temperatures from these regional results to understand the overall temperature range. How can she use parallel computing with MPI to aggregate the local min/max values into global results?

*Answer:*  
Each MPI process computes its local minimum and maximum temperatures. Then, using MPI_Reduce—with the MPI_MIN operator to determine the global minimum and the MPI_MAX operator for the global maximum—the results are aggregated at the root process, which then displays the overall temperature range.

*Descriptive Pseudocode:*

```
// In each MPI process:
function aggregateTemperatureExtremes():
    MPI_Init()                                  // Initialize MPI
    localMin = computeLocalMinTemperature()      // Compute local minimum temperature
    localMax = computeLocalMaxTemperature()      // Compute local maximum temperature
    globalMin = MPI_Reduce(localMin, operation=MIN, root=0)  // Global minimum across processes
    globalMax = MPI_Reduce(localMax, operation=MAX, root=0)  // Global maximum across processes
    if rank == 0:
         print("Global Temperature Range: Min =", globalMin, "Max =", globalMax)
    MPI_Finalize()                              // Finalize MPI
```

*(Reference: citeturn2file19)*

────────────────────────────
**Question 14: QuickBuy Simulation Using Pthreads**

*Question:*  
A popular online shopping platform, “QuickBuy”, has a limited-time offer on a bestselling product. The offer allows only 10 customers to buy the product at a discounted price. The platform’s developer, Alex, wants to ensure that only 10 customers can take advantage of the offer. Alex decides to use multithreading with Pthreads to handle simultaneous requests from customers.  
• Write a program using Pthreads to simulate the scenario where more than 10 customers (threads) try to buy the product, but only 10 can succeed.  
• Compare the results using busy waiting and mutex synchronization. Also, explain each step in the program.

*Answer:*  
Two versions are provided. In the busy waiting version, each thread continuously checks a shared counter without protection, which may lead to race conditions. In the mutex-synchronized version, a mutex ensures that the check-and-decrement operation on the counter is atomic, guaranteeing that exactly 10 customers succeed.

*Descriptive Pseudocode:*

```
// Global variable:
available = 10  // Maximum number of discounted products

// Version A: Busy Waiting (No Mutex)
function customerThread_Busy():
    id = threadID
    while true:
         if available > 0:
              // Without proper locking, multiple threads might decrement concurrently
              available = available - 1
              print("Customer", id, "succeeded!")
              break
         // Continue checking (busy waiting)

// Version B: Using Mutex Synchronization
mutex = new Mutex()  // Create a mutex for protecting shared access
function customerThread_Mutex():
    id = threadID
    lock(mutex)       // Enter critical section
    if available > 0:
         available = available - 1  // Atomically decrement the counter
         print("Customer", id, "succeeded!")
    else:
         print("Customer", id, "failed to buy.")
    unlock(mutex)     // Exit critical section

// Main function:
function main():
    // Create 20 customer threads
    for id from 1 to 20:
         createThread(customerThread)  // Choose either Busy or Mutex version
    // Wait for all threads to finish
    wait for all threads to finish
    print("Total successful sales:", MAX_SALES - available)
```

*(Reference: citeturn2file18 and citeturn2file? – details from the truncated QuickBuy code)*

────────────────────────────
**Question 15: Multithreaded Taylor Series Approximation of sin(x)**

*Question:*  
The Taylor series expansion of sin(x) is given by:  
  sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + x⁹/9! - x¹¹/11! + …  
Design and implement a multithreaded program using Pthreads, mutexes, and semaphores to compute the Taylor series approximation of sin(x).

*Answer:*  
In this design, one thread is spawned for each term in the series (up to a specified number of terms). Each thread computes its own term (using the formula with alternating signs), and a mutex protects the update of a shared global sum. A semaphore is used to signal the main thread when all threads have finished their computation.

*Descriptive Pseudocode:*

```
// Global variables:
global_sum = 0.0         // Accumulates the sum of computed terms
x = user_provided_value  // Angle in radians for sin(x)
NUM_TERMS = 6            // Number of terms to compute
mutex = new Mutex()      // Mutex to protect global_sum
sem = new Semaphore(0)   // Semaphore to signal thread completion

// Helper function to compute factorial iteratively:
function factorial(n):
    result = 1.0
    for i from 2 to n:
         result = result * i
    return result

// Thread function: each thread computes one term of the series
function compute_term(term_index):
    // Compute the exponent: 2 * term_index + 1
    power = 2 * term_index + 1
    // Compute term = (x^power) / (power!) with sign alternating
    term_value = pow(x, power) / factorial(power)
    if term_index is odd:
         term_value = -term_value  // Alternate the sign
    // Protect global update with mutex
    lock(mutex)
    global_sum = global_sum + term_value
    unlock(mutex)
    sem_post(sem)          // Signal that this thread is done

// Main function:
function main():
    x = readInput()        // Read angle x in radians
    initialize mutex and sem
    for term_index from 0 to NUM_TERMS - 1:
         createThread(compute_term, argument = term_index)
    // Wait for all threads to finish:
    for i from 0 to NUM_TERMS - 1:
         sem_wait(sem)
    // Optionally join threads for clean termination
    print("The Taylor series approximation of sin(x) is:", global_sum)
    destroy mutex and sem
```

*(Reference: citeturn2file9)*

────────────────────────────
**Question 16: OpenMP Producer-Consumer Text Processing**

*Question:*  
Use OpenMP to implement a producer-consumer program where some threads act as producers and others as consumers. Producers read text from separate input files and insert lines into a shared queue. Consumers retrieve these lines, tokenize them into words based on whitespace, and write each word to stdout. Ensure proper synchronization using OpenMP constructs to prevent race conditions while accessing the shared queue. Optimize performance by balancing the workload among producers and consumers. Analyze execution time for different numbers of threads and discuss the impact of parallelism. How would you efficiently implement this producer-consumer text processing system using OpenMP while ensuring synchronization and optimal performance?

*Answer:*  
This design uses a shared circular buffer as a queue protected by an OpenMP lock. Producer threads read lines from their designated files and enqueue them. Consumer threads dequeue lines, tokenize them using whitespace as delimiters, and print the tokens. A shared flag indicates when all producers are finished so that consumers exit when the queue is empty. Performance can be measured using omp_get_wtime() and experimenting with different thread counts.

*Descriptive Pseudocode:*

```
// Define a shared circular queue structure:
struct Queue:
    lines      // Array of string pointers
    front      // Index of the first element
    rear       // Next insertion index
    count      // Number of items currently in the queue
    capacity   // Maximum capacity of the queue
    lock       // OpenMP lock for synchronization

// Function to create and initialize the queue:
function createQueue(capacity):
    q = new Queue
    q.lines = allocate array of capacity
    q.capacity = capacity
    q.count = 0; q.front = 0; q.rear = 0
    omp_init_lock(q.lock)
    return q

// Producer function:
function producer(fileName, queue):
    // Open file and read each line:
    while line = readLineFromFile(fileName):
         omp_set_lock(queue.lock)
         if queue.count < queue.capacity:
              queue.lines[queue.rear] = duplicate(line)
              queue.rear = (queue.rear + 1) mod queue.capacity
              queue.count++ 
         omp_unset_lock(queue.lock)
    // Signal completion (update a shared producers_done counter, atomically)

// Consumer function:
function consumer(queue):
    while true:
         omp_set_lock(queue.lock)
         if queue.count > 0:
              line = queue.lines[queue.front]
              queue.front = (queue.front + 1) mod queue.capacity
              queue.count--
              omp_unset_lock(queue.lock)
              // Tokenize the line:
              tokens = tokenize(line)  // Using whitespace as delimiter
              for each token in tokens:
                   omp_set_lock(output_lock)  // Protect printing
                   print(token)
                   omp_unset_lock(output_lock)
              free(line)
         else:
              // If producers are done and queue is empty, exit:
              if producers_done flag is set:
                   omp_unset_lock(queue.lock)
                   break
              omp_unset_lock(queue.lock)
              sleep(shortDelay)
```

In the main function, set up a parallel region with a mix of producer and consumer threads, initialize the shared queue and synchronization primitives, and measure execution time using omp_get_wtime().

*(Reference: citeturn2file8)*

────────────────────────────
**Question 17: Monte Carlo Estimation of π using OpenMP**

*Question:*  
The Monte Carlo method can be used to estimate the value of π by simulating random dart throws on a square dartboard of side length 2 units. A circle of radius 1 is inscribed within this square, meaning the ratio of darts landing inside the circle to the total number of darts thrown should approximate π/4. The formula for estimating π is:  
  π ≈ 4 × (number of darts inside the circle / total number of darts thrown)  
Each dart is thrown randomly within the square by generating two random numbers x and y in the range [-1, 1] representing the dart’s coordinates. If the dart lands inside the circle (x² + y² ≤ 1), it is counted as a hit.  
Task:  
Implement using OpenMP that uses the Monte Carlo method to estimate π:  
1. Read the total number of dart tosses before parallelizing the computation.  
2. Use OpenMP parallelization to distribute the dart tosses among multiple threads.  
3. Apply the OpenMP reduction clause to accumulate the total number of darts landing inside the circle safely.  
4. Print the final estimate of π after all threads complete their computation.

*Answer:*  
This OpenMP-based C++ program first reads the total number of tosses. It then uses a parallel for-loop (with a reduction clause) to simulate dart throws. Each thread generates random x and y coordinates in [-1, 1] and increments a shared counter if the dart lands inside the circle. Finally, π is estimated using the formula provided.

*Descriptive Pseudocode:*

```
// Main function:
function monteCarloPi():
    totalTosses = readInput("Enter total number of dart tosses: ")
    insideCircle = 0
    // Parallelize the dart tosses using OpenMP:
    #pragma omp parallel
    {
        // Initialize a thread-local random seed:
        seed = cast_to_unsigned(currentTime()) + omp_get_thread_num()
        // Use a parallel for-loop with reduction to sum insideCircle:
        #pragma omp for reduction(+:insideCircle)
        for i from 0 to totalTosses - 1:
              // Generate random coordinates between -1 and 1:
              x = (rand_r(&seed) / RAND_MAX) * 2.0 - 1.0
              y = (rand_r(&seed) / RAND_MAX) * 2.0 - 1.0
              if (x*x + y*y <= 1.0):
                   insideCircle++  // Count the hit if inside the circle
    }
    // Estimate π:
    pi_estimate = 4.0 * insideCircle / totalTosses
    print("Estimated value of π:", pi_estimate)
```

*(Reference: citeturn2file13)*

────────────────────────────
This complete document covers all 17 questions from the assignment. Each section provides the full question details, a brief theoretical explanation (where appropriate), and detailed pseudocode with descriptive comments to aid understanding. You can use this text to generate your PDF or as a study guide.

Let me know if you need any further modifications or clarifications.

