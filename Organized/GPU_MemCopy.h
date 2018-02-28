
__global__ void 
MemCopy_kernel ( const int n , const int numBlock, 
                         const float* __restrict__ input , 
                         const float* __restrict__ sol , 
                         const float* __restrict__ x_k , 
                         float* __restrict__ x_k1)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx >= n ) 
        return ;
    float t = 0 ;
    for ( int j = 0 ; j < n ; j++ ){
        if( j != idx ){
            t += input[idx*n+j]*x_k[j] ; 
        }
    }
    x_k1[idx] = ( sol[idx] - t ) / (input[idx*n+idx]) ;  
    return ; 
}




void GPU_MemCopy( int n , int iter , float* input , float* sol , float* x_k , float* x_k1 ){


    extern double exeTime ;  
    float* d_xk  ; //= new float[*n]; 
    float* d_xk1 ; //= new float[*n]; 

    cudaMalloc((void**)&d_xk , n*sizeof(float)) ; 
    cudaMalloc((void**)&d_xk1, n*sizeof(float)) ; 

    cudaMemcpy( d_xk , x_k , n*sizeof(float) , cudaMemcpyHostToDevice);

    float *d_input, *d_sol ; 
    cudaMalloc((void**)&d_input  , n*n*sizeof(float)) ; 
    cudaMalloc((void**)&d_sol    , n*sizeof(float))   ; 
    cudaMemcpy( d_input , input  , n*n*sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy( d_sol   , sol    , n*sizeof(float)   , cudaMemcpyHostToDevice);
    
    const int numBlock  = 160 ; 
    const int blocksize = 32 ; 
    clock_t c_start = clock();
    for ( int it = 0 ; it < iter ; it++){
        printf( "iter = %d \n" , it ) ;
        MemCopy_kernel<<< numBlock , blocksize  >>> ( n , numBlock , d_input , d_sol , d_xk , d_xk1 ) ;
        cudaDeviceSynchronize(); 
        swap_pointer( d_xk , d_xk1 ) ; 
    }
    clock_t c_end = clock();
    exeTime = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;


    cudaMemcpy( x_k   , d_xk   , n*sizeof(float) , cudaMemcpyDeviceToHost);


}
