
__global__ void 
Reduce_kernel ( const int n , const int numBlock, 
                         const float* __restrict__ input , 
                         const float* __restrict__ sol , 
                         const float* __restrict__ x_k , 
                         float* __restrict__ x_k1)
{
    //const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int bid = blockIdx.x  ;
    const int tid = threadIdx.x ;

    extern __shared__ float s_data[];
    //__syncthreads();
    //extern __shared__ int s_xk[1024];
    s_data[ tid ] =  input[ bid*n + tid  ] * x_k[ tid ] 
      + input[ bid*n + tid + blockDim.x ] * x_k[ tid + blockDim.x]  ;
    //__syncthreads();
    //s_data[ bid ] = 0 ; 
    //s_xk[ tid ]    =  x_k[ tid ] ;   
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



void GPU_Reduction( int n , int iter , float* input , float* sol , float* x_k , float* x_k1 ){


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
    
    const int blocksize = 128 ; 
    const int numBlock = n / blocksize ; 
    
    for ( int it = 0 ; it < iter ; it++){
        printf( "iter = %d \n" , it ) ;
        Reduce_kernel<<< n , n/2 , (n/2)*sizeof(float) >>> ( n , numBlock , d_input , d_sol , d_xk , d_xk1 ) ;
        cudaDeviceSynchronize(); 
        swap_pointer( d_xk , d_xk1 ) ; 
    }


    cudaMemcpy( x_k   , d_xk   , n*sizeof(float) , cudaMemcpyDeviceToHost);


}
