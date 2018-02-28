
__global__ void 
Shared_kernel ( const int n , const int numBlock, 
                         const float* __restrict__ input , 
                         const float* __restrict__ sol , 
                         const float* __restrict__ x_k , 
                         float* __restrict__ x_k1)
{
    //__shared__ float s_x_k[ 5120 ] ; 
    //const int sh_size = 5120 ; //(n > 5120)? 5120 : n  ; 
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if( idx >= n ) 
        return ;

    float t = 0 ;
    //printf( " blockIdx.x = %d , blockDIm.x = %d , threadIdx.x = %d \n" , blockIdx.x , blockDim.x , threadIdx.x ) ;
    //const int shared_idx = threadIdx.x * numBlock ; 
    //const int sh_size = 512 ;  
    //const int n_shIter = sh_size / blockDim.x ; 

    //for( int k = 0 ; k < 1 ; k ++){
   //     for( int i = 0 ; i < n_shIter  ; i++){
   //         s_x_k[ ( threadIdx.x * n_shIter ) + i ] = x_k [ ( threadIdx.x * n_shIter  ) + /*k * sh_size + */i ]  ;
            //s_x_k[  threadIdx.x + blockDim.x * i ] = x_k [ threadIdx.x + blockDim.x * i  ]  ;
   //     }
   //     __syncthreads();
        for ( int j = 0 ; j < n ; j++ ){
            if( ( /*k * sh_size + */ j) != idx ){
                //t += input[ idx*n  /*+ k*sh_size */ + j ]*s_x_k[ j ] ; 
                //t += input[ j*n + idx ]*s_x_k[ j ] ; 
                t += input[ j*n + idx ]*x_k[ j ] ; 
                //t += s_x_k[ j ] ; 
            }
        }
    //    __syncthreads();
    //}//__syncthreads();

    x_k1[idx] = ( sol[idx] - t ) / (input[idx*n+idx]) ;  

    return ; 
}


void GPU_Shared( int n , int iter , float* input , float* sol , float* x_k , float* x_k1 ){


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
        Shared_kernel<<< numBlock , blocksize  >>> ( n , numBlock , d_input , d_sol , d_xk , d_xk1 ) ;
        printf( "iter = %d \n" , it ) ;
        cudaDeviceSynchronize(); 
        swap_pointer( d_xk , d_xk1 ) ; 
    }

    cudaMemcpy( x_k   , d_xk   , n*sizeof(float) , cudaMemcpyDeviceToHost);

}
