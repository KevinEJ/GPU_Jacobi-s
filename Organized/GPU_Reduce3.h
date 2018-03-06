template <unsigned int blockSize> 
__global__ void 
Reduce3_kernel ( const int n , const int n_load, 
                         const float* __restrict__ input , 
                         const float* __restrict__ sol , 
                         const float* __restrict__ x_k , 
                         float* __restrict__ x_k1)
{
    //const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int bid = blockIdx.x  ;
    const int tid = threadIdx.x ;
    const int idx = bid * n ;
    extern __shared__ float s_data[];
    if( tid >= n ) return ; 
    
    //s_data[ tid ] = input[ bid * n +  tid  ] * x_k[ tid ] ; 
    s_data[ tid ] = input[ idx +  tid  ] * x_k[ tid ] ; 

    #pragma unroll
    for( int i = 1 ; i < n_load ; i++){
        //s_data[ tid ] += input[ bid * n + i * blockSize + tid  ] * x_k[ i * blockSize + tid ] ; 
        s_data[ tid ] += input[ idx + i * blockSize + tid  ] * x_k[ i * blockSize + tid ] ; 
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

    
    //for (unsigned int s = blockDim.x/2 ; s > 32 ; s>>=1 ) {
    //    if (tid < s) { s_data[tid] += s_data[tid + s]; } __syncthreads();
    //}
    
    if( blockSize >= 1024 ){
        if (tid < 512) { s_data[tid] += s_data[tid + 512]; } __syncthreads();
    }
    
    if( blockSize >= 512 ){
        if (tid < 256) { s_data[tid] += s_data[tid + 256]; } __syncthreads();
    }
    if( blockSize >= 256 ){
        if (tid < 128) { s_data[tid] += s_data[tid + 128]; } __syncthreads();
    }
    if( blockSize >= 128 ){
        if (tid < 64) { s_data[tid] += s_data[tid + 64]; } __syncthreads();
    }

    if (tid < 32)
    {
        if (blockSize >= 64) 
        s_data[tid] += s_data[tid + 32];
        __syncthreads();
        if (blockSize >= 32) 
        s_data[tid] += s_data[tid + 16];
        __syncthreads();
        if (blockSize >= 16) 
        s_data[tid] += s_data[tid + 8];
        __syncthreads();
        if (blockSize >= 8) 
        s_data[tid] += s_data[tid + 4];
        __syncthreads();
        if (blockSize >= 4) 
        s_data[tid] += s_data[tid + 2];
        __syncthreads();
        if (blockSize >= 2) 
        s_data[tid] += s_data[tid + 1];
    }

    if( tid == 0 ){
        //s_data[0] -= input[bid*n+bid] * x_k[ bid ] ; 
        s_data[0] -= input[ idx +bid] * x_k[ bid ] ; 
        //x_k1[bid] = ( sol[bid] - s_data[0] ) / (input[ bid*n + bid ]) ;  
        x_k1[bid] = ( sol[bid] - s_data[0] ) / (input[ idx + bid ]) ;  
    }
    return ; 
}



void GPU_Reduction3( int n , int iter , float* input , float* sol , float* x_k , float* x_k1 ){

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    float* d_xk  ; //= new float[*n]; 
    float* d_xk1 ; //= new float[*n]; 
    
    extern double exeTime ;  


    cudaMalloc((void**)&d_xk , n*sizeof(float)) ; 
    cudaMalloc((void**)&d_xk1, n*sizeof(float)) ; 

    //for( int i = 0 ; i < n ; i++)
    //    x_k[i] = 0 ;
    
    cudaMemcpy( d_xk , x_k , n*sizeof(float) , cudaMemcpyHostToDevice);

    float *d_input, *d_sol ; 
    cudaMalloc((void**)&d_input  , n*n*sizeof(float)) ; 
    cudaMalloc((void**)&d_sol    , n*sizeof(float))   ; 
    cudaMemcpy( d_input , input  , n*n*sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy( d_sol   , sol    , n*sizeof(float)   , cudaMemcpyHostToDevice);
    
    extern int g_Block_size ; 
    //const int blocksize = g_Block_size ; 
    const int blocksize = 512 ; 
    // const int numBlock = n ;
    //const int n_load = n / blocksize ; 
    const int n_load = ((n / blocksize)==0)? 1: n/blocksize ; 
    
    clock_t c_start = clock();
    for ( int it = 0 ; it < iter ; it++){
        printf( "iter = %d \n" , it ) ;
        switch (blocksize)
        {
            case 1024:
                Reduce3_kernel<1024><<< n , blocksize , blocksize*sizeof(float) >>> 
                            ( n , n_load , d_input , d_sol , d_xk , d_xk1 ) ; break ;
            case 512:
                Reduce3_kernel<512><<< n , blocksize , blocksize*sizeof(float) >>> 
                            ( n , n_load , d_input , d_sol , d_xk , d_xk1 ) ; break ;
            case 256:
                Reduce3_kernel<256><<< n , blocksize , blocksize*sizeof(float) >>> 
                            ( n , n_load , d_input , d_sol , d_xk , d_xk1 ) ; break ;
            case 128:
                Reduce3_kernel<128><<< n , blocksize , blocksize*sizeof(float) >>> 
                            ( n , n_load , d_input , d_sol , d_xk , d_xk1 ) ; break ;
            case 64:
                Reduce3_kernel<64><<< n , blocksize , blocksize*sizeof(float) >>> 
                            ( n , n_load , d_input , d_sol , d_xk , d_xk1 ) ; break ;
            case 32:
                Reduce3_kernel<32><<< n , blocksize , blocksize*sizeof(float) >>> 
                            ( n , n_load , d_input , d_sol , d_xk , d_xk1 ) ; break ;
            case 16:
                Reduce3_kernel<16><<< n , blocksize , blocksize*sizeof(float) >>> 
                            ( n , n_load , d_input , d_sol , d_xk , d_xk1 ) ; break ;
            case 8:
                Reduce3_kernel<8><<< n , blocksize , blocksize*sizeof(float) >>> 
                            ( n , n_load , d_input , d_sol , d_xk , d_xk1 ) ; break ;
            case 4:
                Reduce3_kernel<4><<< n , blocksize , blocksize*sizeof(float) >>> 
                            ( n , n_load , d_input , d_sol , d_xk , d_xk1 ) ; break ;
            case 2:
                Reduce3_kernel<2><<< n , blocksize , blocksize*sizeof(float) >>> 
                            ( n , n_load , d_input , d_sol , d_xk , d_xk1 ) ; break ;
            case 1:
                Reduce3_kernel<1><<< n , blocksize , blocksize*sizeof(float) >>> 
                            ( n , n_load , d_input , d_sol , d_xk , d_xk1 ) ; break ;
        }
        //Reduce2_kernel<<< numBlock , blocksize , blocksize*sizeof(float) >>> ( n , numBlock , d_input , d_sol , d_xk , d_xk1 ) ;
        cudaDeviceSynchronize(); 
        swap_pointer( d_xk , d_xk1 ) ; 
    }
    clock_t c_end = clock();
    exeTime = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;

    cudaMemcpy( x_k   , d_xk   , n*sizeof(float) , cudaMemcpyDeviceToHost);


}
