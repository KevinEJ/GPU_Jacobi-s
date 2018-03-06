
__global__ void 
Reduce_kernel ( const int n , const int n_load, 
                         const float* __restrict__ input , 
                         const float* __restrict__ sol , 
                         const float* __restrict__ x_k , 
                         float* __restrict__ x_k1)
{
    //const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int bid = blockIdx.x  ;
    const int tid = threadIdx.x ;
    const int idx = bid * n ;
    const int blockSize = blockDim.x ;

    extern __shared__ float s_data[];
    //__syncthreads();
    //extern __shared__ int s_xk[1024];
    if( tid >= n ) return ; 
    
    //s_data[ tid ] = input[ bid * n +  tid  ] * x_k[ tid ] ; 
    s_data[ tid ] = input[ idx +  tid  ] * x_k[ tid ] ; 

    for( int i = 1 ; i < n_load ; i++){
        //s_data[ tid ] += input[ bid * n + i * blockSize + tid  ] * x_k[ i * blockSize + tid ] ; 
        s_data[ tid ] += input[ idx + i * blockSize + tid  ] * x_k[ i * blockSize + tid ] ; 
    }
    
    __syncthreads();

    //float t = 0 ;
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if( tid == 0 ){
        s_data[0] -= input[bid*n+bid] * x_k[ bid ] ; 
        x_k1[bid] = ( sol[bid] - s_data[0] ) / (input[ bid*n + bid ]) ;  
    }
    return ; 
}



void GPU_Reduction( int n , int iter , float* input , float* sol , float* x_k , float* x_k1 ){


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
    
    extern int g_Block_size ; 
    //const int blocksize = g_Block_size ; 
    const int blocksize = 1024 ; 
    //const int n_load = n / blocksize ; 
    const int n_load = ((n / blocksize)==0)? 1: n/blocksize ; 
    
    clock_t c_start = clock();
    for ( int it = 0 ; it < iter ; it++){
        printf( "iter = %d \n" , it ) ;
        // Reduce_kernel<<< n , n/2 , (n/2)*sizeof(float) >>> ( n , n_load , d_input , d_sol , d_xk , d_xk1 ) ;
        Reduce_kernel<<< n , blocksize , blocksize*sizeof(float) >>> 
                            ( n , n_load , d_input , d_sol , d_xk , d_xk1 ) ; 
        cudaDeviceSynchronize(); 
        swap_pointer( d_xk , d_xk1 ) ; 
    }

    clock_t c_end = clock();
    exeTime = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;

    cudaMemcpy( x_k   , d_xk   , n*sizeof(float) , cudaMemcpyDeviceToHost);


}
