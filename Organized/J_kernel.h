

__global__ void 
J_kernel ( const int n , const int numBlock, 
                         const float* __restrict__ input , 
                         const float* __restrict__ sol , 
                         const float* __restrict__ x_k , 
                         float* __restrict__ x_k1)
{
   // __shared__ float s_sol[5120] ; 
    //__shared__ volatile float s_x_k[5120] ; 
    __shared__ float s_x_k[512] ; 
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if( idx >= n ) 
        return ;

    float t = 0 ;
    //printf( " blockIdx.x = %d , blockDIm.x = %d , threadIdx.x = %d \n" , blockIdx.x , blockDim.x , threadIdx.x ) ;
    const int shared_idx = threadIdx.x * numBlock ; 
   
    //const int sh_size = 512 ;  

    for( int k = 0 ; k < 10 ; k ++){
        for( int i = 0 ; i < 8  ; i++){
        //s_x_k[ (threadIdx.x * blockDim.x) + i ] = x_k [ (threadIdx.x * blockDim.x) + i ]  ;
        //s_x_k[ (threadIdx.x * 64 ) + i ] = x_k [ (threadIdx.x * 64 ) + i ]  ;
        //s_x_k[ ( shared_idx ) + i ] = x_k [ (shared_idx ) + i ]  ;
            s_x_k[ ( threadIdx.x * 8 ) + i ] = x_k [ ( threadIdx.x * 8  ) + k*512 + i ]  ;
        }
        //}
        __syncthreads();

        for ( int j = 0 ; j < 512 ; j++ ){
            //int = 512 * j ; 
            if( (k* 512 + j) != idx ){
                t += input[idx*n + k*512 + j]*s_x_k[ j ] ; 
            }
        }
        __syncthreads();
    }//__syncthreads();

    x_k1[idx] = ( sol[idx] - t ) / (input[idx*n+idx]) ;  
    //x_k1[idx] =  (input[idx][idx]) ;
    //float aaa = sol[idx] ;
    //x_k1[idx] = aaa   ;  
    
    // swap x_k and x_k1

    //x_k1[idx] = 3.0 ; // = sol[idx] ;
    //x_k[idx]  = 4.0 ; //= sol[idx] ;

    return ; 
}
