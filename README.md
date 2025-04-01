Below is a revised version of the answers. The pseudocode has been expanded with extra comments and plain language explanations so that even a layperson can follow the logic.

────────────────────────────
**Question 1: Approaches to Writing Parallel Programs and MIMD Systems**

*Question:*  
In modern computing, performance improvements rely heavily on parallel programming techniques. Describe different approaches to writing parallel programs, providing examples of each. Additionally, explain how parallel systems are classified and differentiate them from concurrent and distributed systems. How do the principal types of MIMD systems contribute to efficient parallel computing?

*Answer (Theoretical):*  
Parallel programming uses various methods:
- **Thread-Based (Shared Memory):** Multiple threads run simultaneously within the same memory space (e.g., using OpenMP or Pthreads).
- **Message Passing (Distributed Memory):** Independent processes exchange messages (e.g., via MPI).
- **Data Parallelism:** The same operation is applied simultaneously on different pieces of data.
- **Task Parallelism:** Different tasks are executed concurrently.

MIMD systems allow different processors to execute different instructions on different data, which increases flexibility and scalability.

*Detailed Pseudocode:*

```
// Example: Thread-Based Parallelism (Shared Memory)
// This pseudocode explains how to create multiple threads that perform a simple task.
function threadParallelExample():
    // Imagine we have a pool of threads that we want to run simultaneously.
    for each thread in THREAD_POOL:
         // The 'parallel_execute' block means these commands run at the same time.
         parallel_execute:
              // Each thread prints a greeting along with its unique identifier.
              print("Hello from thread", thread.id)

// Example: Message Passing (Distributed Memory)
// This pseudocode shows the basic structure of an MPI program where processes communicate.
function mpiExample():
    // Initialize the MPI environment for communication.
    MPI_Init()
    // Determine the unique identifier (rank) for the current process.
    rank = MPI_Comm_rank()
    // Check if this is the master process (rank 0).
    if rank == 0:
         // The master process prints a message.
         print("Hello from process", rank)
    else:
         // Other processes perform their specific tasks.
         performLocalWork()
    // Close the MPI environment after the work is done.
    MPI_Finalize()
```

────────────────────────────
**Question 2: Von Neumann Architecture and Memory Models**

*Question:*  
Explain the Von Neumann architecture and its significance in modern computing. How does it relate to parallel and distributed systems? Additionally, compare shared memory and distributed memory MIMD systems.

*Answer:*  
The Von Neumann architecture uses a single memory for both instructions and data. This design simplifies programming but may cause delays because the CPU has to share one bus for both data and instructions. Modern systems use multiple cores and parallel processing to overcome these limitations. In shared memory systems, all cores access the same memory, whereas in distributed memory systems, each core has its own memory and they communicate by sending messages.

*Detailed Pseudocode:*

```
// Example: Von Neumann Execution Cycle
// This pseudocode represents the basic operation of a computer using the Von Neumann model.
function vonNeumannCycle():
    // Load both the program and the necessary data from memory.
    load_program_and_data_from_memory()
    // Continue processing until the end of the program.
    while not end_of_program:
         // Fetch the next instruction from memory.
         instruction = fetch_from_memory()
         // Decode the fetched instruction.
         decoded = decode(instruction)
         // Execute the decoded instruction.
         execute(decoded)
```

────────────────────────────
**Question 3: Cloud-Based AI Training Platform**

*Question:*  
A cloud-based AI training platform is designed to process large-scale machine learning workloads by distributing computations across multiple independent servers, each with its own private memory. The system must efficiently exchange model parameters between nodes while minimizing communication overhead. In this scenario, which type of MIMD system would be ideal, and how does it handle data exchange between processors?

*Answer:*  
A distributed memory MIMD system is ideal for this scenario. Each server computes its own model update, and then they use message-passing protocols (such as MPI_Allreduce) to efficiently exchange and aggregate these updates, ensuring that all nodes are synchronized.

*Detailed Pseudocode:*

```
// Example: Distributed AI Training using MPI
function distributedAITraining():
    // Start the MPI environment to enable communication between different servers.
    MPI_Init()
    // Retrieve the local data for model training specific to this server.
    local_data = getLocalData()
    // Compute model updates (e.g., gradients) based on the local data.
    local_update = computeLocalModelUpdate(local_data)
    // Use MPI_Allreduce to combine all local updates into one global update.
    // This operation adds up the updates from all servers and distributes the result.
    global_update = MPI_Allreduce(local_update, operation=SUM)
    // Update the overall model with the aggregated global update.
    updateModel(global_update)
    // Finalize the MPI environment after completing the operation.
    MPI_Finalize()
```

────────────────────────────
**Question 4: Pthreads Dynamic Task Queue**

*Question:*  
You are tasked with developing a multi-threaded system using Pthreads to manage a dynamic task queue. The program should start with a user-specified number of worker threads that initially sleep in a condition wait. The main thread generates blocks of tasks and wakes a worker thread with a condition signal. After completing its tasks, the worker returns to waiting. Once all tasks are generated, the main thread signals completion and uses a condition broadcast to wake all threads for termination. Implement the system with proper synchronization.

*Answer:*  
The solution uses a shared task queue protected by a mutex. Worker threads wait for tasks using a condition variable. When new tasks are available, one worker is signaled to wake up. Once task generation is complete, all workers are notified so they can finish.

*Detailed Pseudocode:*

```
// Global variables are shared between threads.
taskQueue = empty queue           // Holds the tasks to be processed.
mutex = new Mutex()               // Ensures that only one thread accesses the queue at a time.
conditionVar = new ConditionVariable()  // Allows threads to wait until new tasks are added.
tasksDone = false                 // Flag indicating that no more tasks will be added.

// Function to add a new task to the queue.
function enqueueTask(task):
    lock(mutex)                   // Protect the queue with a lock.
    taskQueue.enqueue(task)       // Add the new task.
    signal(conditionVar)          // Notify one waiting worker that a task is available.
    unlock(mutex)                 // Release the lock.

// Function that defines what each worker thread does.
function workerThread(threadId):
    while true:
         lock(mutex)
         // If the task queue is empty and there are still tasks expected, wait.
         while taskQueue.isEmpty() and not tasksDone:
              wait(conditionVar, mutex)
         // If the queue is empty and no more tasks will come, exit the loop.
         if taskQueue.isEmpty() and tasksDone:
              unlock(mutex)
              break
         // Otherwise, remove a task from the queue.
         task = taskQueue.dequeue()
         unlock(mutex)
         // Process the task (this is where the actual work is done).
         process(task)
         sleep(1)               // Simulate time taken to process.
    print("Thread", threadId, "exiting")
```

────────────────────────────
**Question 5: Handling Imbalanced Workload in Matrix-Vector Multiplication**

*Question:*  
Imagine you are implementing a parallel matrix-vector multiplication algorithm for a large dataset. Initially, the matrix dimensions are perfectly divisible by the number of available threads. However, if the dimensions do not divide evenly, describe an approach to handle such an imbalanced workload efficiently.

*Answer:*  
First, calculate the number of rows each thread should process using integer division. Then, assign the extra rows (the remainder) to the first few threads. This ensures each thread gets as balanced a workload as possible.

*Detailed Pseudocode:*

```
// 'm' is the total number of rows and 't' is the total number of threads.
function assignRows(m, t):
    base = m / t              // Compute the basic number of rows per thread.
    remainder = m mod t       // Extra rows that do not evenly divide.
    // For each thread, compute the starting row and number of rows to process.
    for i from 0 to t - 1:
         // Each thread gets the base number of rows, plus one extra if its index is less than the remainder.
         start = i * base + min(i, remainder)
         if i < remainder:
             count = base + 1  // This thread gets an extra row.
         else:
             count = base
         end = start + count - 1  // The last row index for this thread.
         // Assign the task to the thread.
         assignTaskToThread(i, start, end)
```

────────────────────────────
**Question 6: OpenMP Histogram Program**

*Question:*  
Develop an OpenMP program to implement a histogram where the master thread receives sensor data in the range 0–99, and slave threads (each representing a bucket) count the data falling within their range. The master thread displays bucket counts periodically.

*Answer:*  
The master thread simulates sensor data and updates a shared data array. Each worker thread (representing a histogram bucket) continuously checks for new data and updates its count using atomic operations to ensure correctness. The master periodically prints the current counts.

*Detailed Pseudocode:*

```
// Global variables for shared data.
sensorData = array of size TOTAL_DATA   // To hold sensor values.
bucketCounts = array of size NUM_BUCKETS, initialized to 0
dataIndex = 0   // Indicates how many sensor data items have been produced.

parallel region with NUM_BUCKETS+1 threads:
    threadId = omp_get_thread_num()
    if threadId == 0:
         // Master thread: simulates receiving sensor data.
         for i from 0 to TOTAL_DATA-1:
              sensorData[i] = randomInteger(0, 99)  // Generate a random sensor value.
              dataIndex++                           // Increase the counter.
              if (i mod 100 == 0):
                  print(bucketCounts)               // Periodically display current counts.
              sleep(delay)                          // Simulate delay between data points.
    else:
         // Worker threads: each corresponds to one bucket.
         bucket = threadId - 1
         // Define the range of sensor values for this bucket.
         lowerBound = bucket * (100 / NUM_BUCKETS)
         upperBound = lowerBound + (100 / NUM_BUCKETS)
         myIndex = 0    // Tracks which sensor data items this thread has processed.
         while true:
              localIndex = dataIndex    // Get the latest available index.
              // Process any new sensor data that has arrived.
              while myIndex < localIndex:
                    if sensorData[myIndex] >= lowerBound and sensorData[myIndex] < upperBound:
                         // Atomically increment the bucket count to avoid conflicts.
                         atomic_increment(bucketCounts[bucket])
                    myIndex++  // Move to the next data item.
              if localIndex >= TOTAL_DATA:
                    break    // Exit when all data have been processed.
              sleep(shortDelay)  // Short pause to reduce CPU busy waiting.
```

────────────────────────────
**Question 7: MPI Global Min/Max Shipping Times**

*Question:*  
You are running a large e-commerce operation where each warehouse computes the minimum and maximum shipping times. Using MPI, write a program flow to collect these local min/max values and compute the global minimum and maximum shipping times.

*Answer:*  
Each process computes its local minimum and maximum shipping times. MPI_Reduce is then used with the MPI_MIN operator to gather the smallest value and with MPI_MAX to gather the largest value. The results are available in the root process.

*Detailed Pseudocode:*

```
// Each process follows these steps:
function aggregateShippingTimes():
    MPI_Init()  // Start MPI communication.
    // Each process computes its own local shipping time extremes.
    localMin = computeLocalMinShippingTime()
    localMax = computeLocalMaxShippingTime()
    // Gather the smallest shipping time across all processes.
    globalMin = MPI_Reduce(localMin, operation=MIN, root=0)
    // Gather the largest shipping time across all processes.
    globalMax = MPI_Reduce(localMax, operation=MAX, root=0)
    if rank == 0:
         // Only the root process prints the global results.
         print("Global Min Shipping Time:", globalMin)
         print("Global Max Shipping Time:", globalMax)
    MPI_Finalize()  // End MPI communication.
```

────────────────────────────
**Question 8: Ping-Pong Communication and Timing**

*Question:*  
You implement a ping-pong communication pattern between two processes to measure message passing cost. How long does the program need to run for clock() to report a nonzero runtime? How do timings from clock() compare to those from MPI_Wtime()?

*Answer:*  
The clock() function measures CPU time and typically shows a nonzero value after about 1/CLOCKS_PER_SEC seconds (approximately 1 microsecond). In contrast, MPI_Wtime() measures the wall-clock (elapsed) time, including waiting periods, which is more accurate for communication tests.

*Detailed Pseudocode:*

```
// This pseudocode is for process 0 (the initiator).
function pingPongTest():
    MPI_Init()  // Start MPI communication.
    if rank == 0:
         // Record the start times using both timing functions.
         startWall = MPI_Wtime()  // Wall-clock time.
         startCPU = clock()       // CPU time.
         for count from 0 to LIMIT-1:
              // Send a message to process 1.
              MPI_Send(message, destination=1)
              // Receive a message back from process 1.
              MPI_Recv(message, source=1)
         endCPU = clock()         // Record CPU time after loop.
         endWall = MPI_Wtime()      // Record wall-clock time after loop.
         // Calculate and print the measured times.
         cpuTime = (endCPU - startCPU) / CLOCKS_PER_SEC
         wallTime = endWall - startWall
         print("CPU time (clock):", cpuTime)
         print("Wall-clock time (MPI_Wtime):", wallTime)
    else:
         // Process 1 simply responds to each message.
         for count from 0 to LIMIT-1:
              MPI_Recv(message, source=0)
              MPI_Send(message, destination=0)
    MPI_Finalize()  // End MPI communication.
```

────────────────────────────
**Question 9: Measuring Redistribution Cost Between Block and Cyclic Distributions**

*Question:*  
Imagine a large vector is distributed among processes. Write pseudocode to measure the time cost for redistributing the vector between block and cyclic distributions. Also, discuss factors influencing the redistribution time.

*Answer:*  
The time to redistribute data can be measured by capturing the wall-clock time before and after the transfer. The redistribution time is affected by the vector size, the number of processes, network latency, bandwidth, and overhead in message handling.

*Detailed Pseudocode:*

```
// This pseudocode demonstrates measuring redistribution time.
function measureRedistributionCost():
    // --- Block-to-Cyclic Redistribution ---
    startTime = MPI_Wtime()  // Get the current wall-clock time.
    // For each element in the local block:
    for each local element with global index i:
         // Calculate the target process based on cyclic distribution.
         targetProcess = i mod P
         if targetProcess != myRank:
              // Send the element to the target process.
              MPI_Send(element, targetProcess)
         else:
              // If the element remains in the same process, store it locally.
              cyclicBuffer.add(element)
    // Wait until all processes have received their expected elements.
    while not all expected elements received:
         element = MPI_Recv(anySource)
         cyclicBuffer.add(element)
    blockToCyclicTime = MPI_Wtime() - startTime
    print("Block-to-Cyclic Redistribution Time:", blockToCyclicTime)
    
    // (Similar steps would be repeated for cyclic-to-block redistribution.)
```

────────────────────────────
**Question 10: Thread-Safe Tokenizer Without Modifying the Original String**

*Question:*  
Design a thread-safe tokenizer that extracts tokens from a string without modifying the original input. Provide pseudocode and outline your implementation considerations.

*Answer:*  
The tokenizer maintains its own state (using a pointer) so that the original string is untouched. It returns new copies of each token. Each thread uses its own state variable (savePtr) to ensure thread safety, and memory is allocated for each token which must be freed by the caller.

*Detailed Pseudocode:*

```
// Function to tokenize a string in a thread-safe manner.
function threadSafeTokenizer(input, delimiters, savePtr):
    // On the first call, input is provided. Subsequent calls pass NULL.
    if input is not NULL:
         savePtr = input  // Initialize the pointer to the start of the input.
    // If there is no data left, return NULL.
    if savePtr is NULL or at end of string:
         return NULL
    // Mark the start of the next token.
    start = savePtr
    // Search for the first occurrence of any delimiter character.
    pos = findFirstOccurrence(savePtr, delimiters)
    if pos is NULL:
         // No delimiter found: the rest of the string is the token.
         token = duplicateString(start)
         savePtr = NULL  // No more tokens remain.
    else:
         // Calculate the length of the token.
         tokenLength = pos - start
         // Allocate memory for the token (plus one for the null terminator).
         token = allocateMemory(tokenLength + 1)
         // Copy the token from the original string.
         copy substring from start of length tokenLength into token
         // Ensure the token ends with a null character.
         token[tokenLength] = '\0'
         // Update the pointer to start after the delimiter.
         savePtr = pos + 1
    return token

// Usage example:
function processString(input, delimiters):
    savePtr = NULL
    token = threadSafeTokenizer(input, delimiters, savePtr)
    while token is not NULL:
         // Process the token (e.g., print it or analyze it).
         process(token)
         // Free the token's memory after processing.
         free(token)
         // Get the next token.
         token = threadSafeTokenizer(NULL, delimiters, savePtr)
```

────────────────────────────
**Question 11: MPI Block-Column Distribution for Matrix-Vector Multiplication**

*Question:*  
Describe how you would implement block-column distribution and matrix-vector multiplication in MPI. Include steps for initializing MPI, distributing data, performing local multiplication, reducing results, and finalizing.

*Answer:*  
Process 0 reads the entire matrix and vector, divides the matrix into block columns, and sends each block to the corresponding process. Each process multiplies its submatrix with its corresponding vector segment. The partial results are then combined using MPI_Reduce.

*Detailed Pseudocode:*

```
// Main function for distributed matrix-vector multiplication.
function distributedMatrixVectorMultiply():
    MPI_Init()  // Start MPI communication.
    rank = MPI_Comm_rank()
    size = MPI_Comm_size()
    if rank == 0:
         n = readMatrixOrder()         // Get the matrix size.
         A = readFullMatrix()            // Read the complete matrix.
         x = readVector()                // Read the complete vector.
         // Distribute parts of the matrix to each process.
         for each process i:
              // Compute how many columns process i should handle.
              localCols = computeLocalColumns(n, i, size)
              if i == 0:
                   // Process 0 extracts its own block of columns.
                   localA = extractBlockColumns(A, start=0, count=localCols)
              else:
                   // Pack the appropriate block of columns and send to process i.
                   temp = packBlockColumns(A, start=offset, count=localCols)
                   MPI_Send(temp, destination=i)
         // Broadcast the full vector x to all processes.
         MPI_Bcast(x, root=0)
    else:
         // Other processes receive their portion of the matrix.
         localA = allocateMatrix(n, localCols)
         MPI_Recv(localA, source=0)
         // All processes receive the full vector.
         MPI_Bcast(x, root=0)
    // Extract the relevant part of the vector for local multiplication.
    x_local = extractSubVector(x, offset, localCols)
    // Perform the local matrix-vector multiplication.
    y_local = localMatrixVectorMultiply(localA, x_local)
    // Combine all local results into the final result vector.
    global_y = MPI_Reduce(y_local, operation=SUM, root=0)
    if rank == 0:
         print("Resulting vector y:", global_y)
    MPI_Finalize()  // End MPI communication.
```

────────────────────────────
**Question 12: OpenMP Donation Collection Simulation**

*Question:*  
A community group with 10 volunteers collects donations from 10 city regions. Each volunteer collects a random donation amount. Design an OpenMP program that simulates this process and aggregates the total donation.

*Answer:*  
Each volunteer is represented by a thread that generates a random donation value. The donations are stored in an array and summed at the end to calculate the total funds raised.

*Detailed Pseudocode:*

```
// Global array to store each volunteer's donation.
donations = array of size 10, initialized to 0

// Begin a parallel region with 10 threads (one per volunteer).
parallel region with 10 threads:
    threadId = omp_get_thread_num()  // Each thread gets a unique ID.
    // Create a seed for random number generation (different for each thread).
    seed = currentTime() + threadId
    // Generate a random donation amount (for example, between $0.00 and $1000.99).
    donation = generateRandomDonation(seed)
    // Store the donation in the corresponding position in the array.
    donations[threadId] = donation
    // Print the donation for this volunteer.
    print("Volunteer", threadId, "collected:", donation)

// After the parallel region, calculate the total donation.
total = sum(donations)
print("Total funds raised:", total)
```

────────────────────────────
**Question 13: Aggregating Local Temperature Extremes with MPI**

*Question:*  
Dr. Estelle collects temperature data processed by regional teams. Each team computes local minimum and maximum temperatures. Using MPI, how do you aggregate these into global minimum and maximum values?

*Answer:*  
Each MPI process computes the local minimum and maximum temperatures for its region. Then, MPI_Reduce with the MPI_MIN operator is used to find the overall minimum, and MPI_Reduce with the MPI_MAX operator is used to find the overall maximum. The root process receives and displays the results.

*Detailed Pseudocode:*

```
// In each process:
function aggregateTemperatureExtremes():
    MPI_Init()  // Start MPI communication.
    // Each process calculates its local temperature extremes.
    localMin = computeLocalMinTemperature()
    localMax = computeLocalMaxTemperature()
    // Use MPI_Reduce to find the global minimum temperature.
    globalMin = MPI_Reduce(localMin, operation=MIN, root=0)
    // Use MPI_Reduce to find the global maximum temperature.
    globalMax = MPI_Reduce(localMax, operation=MAX, root=0)
    if rank == 0:
         // The root process prints the aggregated results.
         print("Global Temperature Range: Min =", globalMin, "Max =", globalMax)
    MPI_Finalize()  // End MPI communication.
```

────────────────────────────
**Question 14: QuickBuy Simulation Using Pthreads**

*Question:*  
QuickBuy, an online shopping platform, allows only 10 customers to buy a discounted product. Using multithreading with Pthreads, simulate a scenario where more than 10 customer threads attempt to purchase the product, but only 10 succeed. Compare busy waiting versus mutex synchronization.

*Answer:*  
In the busy waiting version, threads repeatedly check a shared counter without any synchronization, which may lead to race conditions. With mutex synchronization, a mutex lock is used to ensure that only one thread updates the counter at a time, ensuring that only 10 purchases are allowed.

*Detailed Pseudocode:*

```
// Version A: Busy Waiting (No Mutex)
// Global variable to keep track of available discounted products.
available = 10

function customerThread_Busy():
    id = threadID
    // Loop until a purchase is made.
    while true:
         // Check if there are any offers left.
         if available > 0:
              // Decrement the available count (this may lead to race conditions).
              available = available - 1
              print("Customer", id, "succeeded!")
              break  // Exit the loop after a successful purchase.
         // Otherwise, continue checking (busy waiting).

// Version B: Using Mutex Synchronization
// Global variables: available products and a mutex lock.
available = 10
mutex = new Mutex()

function customerThread_Mutex():
    id = threadID
    // Lock the critical section to ensure exclusive access.
    lock(mutex)
    if available > 0:
         available = available - 1
         print("Customer", id, "succeeded!")
    else:
         print("Customer", id, "failed to buy.")
    // Unlock the mutex after the operation.
    unlock(mutex)

// Main function for both versions:
function main():
    // Create 20 customer threads.
    for id from 1 to 20:
         createThread(customerThread)  // Choose either the busy waiting or mutex version.
    // Wait until all threads have completed.
    wait for all threads to finish
```

────────────────────────────
This enhanced version includes more descriptive pseudocode and comments to help a layperson understand the logic behind each solution. You can use this text to generate your PDF or for further study.
