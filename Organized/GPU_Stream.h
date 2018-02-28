
__global__ void 
Stream_kernel ( const int n , int streamIdx, int s ,  
                         const float* __restrict__ input , 
                         const float* __restrict__ sol , 
                         const float* __restrict__ x_k , 
                         float* __restrict__ x_k1)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx >= s ) 
        return ; 
    
    streamIdx = streamIdx * s + idx ;  

    float t = 0 ;
    for( int j = 0 ; j < n ; j++){
        if( j != streamIdx )
            t += input[ idx * n + j ] * x_k[j] ; 
    }
/*
    for ( int j = 0 ; j < n ; j++ ){
        if( j != idx ){
            t += input[idx*n+j]*x_k[j] ; 
        }
    }*/
    x_k1[ idx ] = ( sol[ idx ] - t ) / ( input[ idx * n + streamIdx ] ) ;  
    return ; 
}




void GPU_Stream( int n , int iter , float* input , float* sol , float* x_k , float* x_k1 ){


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
    //cudaMemcpy( d_input , input  , n*n*sizeof(float) , cudaMemcpyHostToDevice);
    //cudaMemcpy( d_sol   , sol    , n*sizeof(float)   , cudaMemcpyHostToDevice);
   
    int s = 256 ; 
    int numStream = n / s ; 
    cudaStream_t *streams = new cudaStream_t[ numStream ];
    for( int i = 0 ; i < numStream ; i ++){
        cudaStreamCreate(&streams[i]);
    }


    for( int i = 0 ; i < numStream ; i++){
        cudaMemcpyAsync(  d_input + i*n*s , input + i*n*s , s*n*sizeof(float) ,  cudaMemcpyHostToDevice,streams[i]);              // H2D
        cudaMemcpyAsync(  d_sol   + i*s      , sol   + i*s      , s*sizeof(float) ,  cudaMemcpyHostToDevice,streams[i]);              // H2D
    }    

    const int numBlock  = 160 ; 
    const int blocksize = 32 ; 
    for ( int it = 0 ; it < iter ; it++){
        printf( "iter = %d \n" , it ) ;
        for( int i = 0 ; i < numStream ; i++){
            Stream_kernel<<< numBlock , blocksize , 0 , streams[i] >>> ( n , i , s , d_input + i*n*s , d_sol + i*s , d_xk , d_xk1 + i*s ) ;
        }
        swap_pointer( d_xk , d_xk1 ) ; 
        cudaDeviceSynchronize(); 
    }
    
   /* 
    for ( int it = 0 ; it < iter ; it++){
            Stream_kernel<<< numBlock , blocksize , 0 , stream[i] >>> ( n , numBlock , d_input , d_sol , d_xk , d_xk1 ) ;
            cudaMemcpyAsync(  d_input + i*n , input + i*n , n*sizeof(float) ,  cudaMemcpyHostToDevice,stream[i]);              // H2D
            cudaMemcpyAsync( pD + i*size,  pH+i*size,size,  cudaMemcpyDeviceToHost,stream[i]);  
        }
    }


    const int numBlock  = 160 ; 
    const int blocksize = 32 ; 
    for ( int it = 0 ; it < iter ; it++){
        printf( "iter = %d \n" , it ) ;
        Stream_kernel<<< numBlock , blocksize  >>> ( n , numBlock , d_input , d_sol , d_xk , d_xk1 ) ;
        cudaDeviceSynchronize(); 
        swap_pointer( d_xk , d_xk1 ) ; 
    }
*/
    cudaMemcpy( x_k   , d_xk   , n*sizeof(float) , cudaMemcpyDeviceToHost);

}
