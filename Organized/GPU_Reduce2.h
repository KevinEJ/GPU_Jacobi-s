__global__ void 
Reduce2_kernel ( const int n , const int numBlock, 
                         const float* __restrict__ input , 
                         const float* __restrict__ sol , 
                         const float* __restrict__ x_k , 
                         float* __restrict__ x_k1)
{
    //const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int bid = blockIdx.x  ;
    const int tid = threadIdx.x ;
    extern __shared__ float s_data[];
    //s_data [ tid ] = 0 ; 
    s_data[ tid ] = input[ bid * n +  tid  ] * x_k[ tid ] ; 
    for( int i = 1 ; i < n / blockDim.x ; i++){
        s_data[ tid ] += input[ bid * n + i * blockDim.x + tid  ] * x_k[ i * blockDim.x + tid ] ; 
    } 
   /*
        + input[ bid * n + 1 * blockDim.x + tid  ] * x_k[ 1 * blockDim.x + tid ]   
        + input[ bid * n + 2 * blockDim.x + tid  ] * x_k[ 2 * blockDim.x + tid ]   
        + input[ bid * n + 3 * blockDim.x + tid  ] * x_k[ 3 * blockDim.x + tid ]   
        + input[ bid * n + 4 * blockDim.x + tid  ] * x_k[ 4 * blockDim.x + tid ] ;  
   *//* 
    s_data[ tid ] += input[ bid * n + 1 * blockDim.x + tid  ] * x_k[ 1 * blockDim.x + tid ] ;  
    s_data[ tid ] += input[ bid * n + 2 * blockDim.x + tid  ] * x_k[ 2 * blockDim.x + tid ] ;  
    s_data[ tid ] += input[ bid * n + 3 * blockDim.x + tid  ] * x_k[ 3 * blockDim.x + tid ] ;  
    s_data[ tid ] += input[ bid * n + 4 * blockDim.x + tid  ] * x_k[ 4 * blockDim.x + tid ] ;  
    */
    //__syncthreads();
    __syncthreads();

    //float t = 0 ;
   /* for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }*/
    for (unsigned int s = blockDim.x/2 ; s > 32 ; s>>=1 ) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    
    if (tid < 32)
    {
        s_data[tid] += s_data[tid + 32];
        __syncthreads();
        s_data[tid] += s_data[tid + 16];
        __syncthreads();
        s_data[tid] += s_data[tid + 8];
        __syncthreads();
        s_data[tid] += s_data[tid + 4];
        __syncthreads();
        s_data[tid] += s_data[tid + 2];
        __syncthreads();
        s_data[tid] += s_data[tid + 1];
    }

    if( tid == 0 ){
        s_data[0] -= input[bid*n+bid] * x_k[ bid ] ; 
        x_k1[bid] = ( sol[bid] - s_data[0] ) / (input[ bid*n + bid ]) ;  
    }
    return ; 
}



void GPU_Reduction2( int n , int iter , float* input , float* sol , float* x_k , float* x_k1 ){


    extern double exeTime ;  
    float* d_xk  ; //= new float[*n]; 
    float* d_xk1 ; //= new float[*n]; 

    cudaMalloc((void**)&d_xk , n*sizeof(float)) ; 
    cudaMalloc((void**)&d_xk1, n*sizeof(float)) ; 

    for( int i = 0 ; i < n ; i++)
        x_k[i] = 0 ;
    
    cudaMemcpy( d_xk , x_k , n*sizeof(float) , cudaMemcpyHostToDevice);

    float *d_input, *d_sol ; 
    cudaMalloc((void**)&d_input  , n*n*sizeof(float)) ; 
    cudaMalloc((void**)&d_sol    , n*sizeof(float))   ; 
    cudaMemcpy( d_input , input  , n*n*sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy( d_sol   , sol    , n*sizeof(float)   , cudaMemcpyHostToDevice);
    
    const int blocksize = 64 ; 
    const int numBlock = n ; 
    clock_t c_start = clock();
    
    for ( int it = 0 ; it < iter ; it++){
        printf( "iter = %d \n" , it ) ;
        Reduce2_kernel<<< numBlock , blocksize , blocksize*sizeof(float) >>> ( n , numBlock , d_input , d_sol , d_xk , d_xk1 ) ;
        cudaDeviceSynchronize(); 
        swap_pointer( d_xk , d_xk1 ) ; 
    }
    clock_t c_end = clock();
    exeTime = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;


    cudaMemcpy( x_k   , d_xk   , n*sizeof(float) , cudaMemcpyDeviceToHost);


}
