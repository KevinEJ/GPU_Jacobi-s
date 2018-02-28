#include "memory.h"


__global__ void 
Unified_kernel ( const int n , const int numBlock, 
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


void GPU_Unified( int n , int iter , float* input , float* sol , float* x_k , float* x_k1 ){
    // Mem Assign 
    extern double exeTime ;  
    float* u_input ; 
    float* u_sol   ; 
    cudaMallocManaged(&u_input, n*n*sizeof(float));
    cudaMallocManaged(&u_sol  ,   n*sizeof(float));

    memcpy( u_input , input , n*n*sizeof(float) ) ; 
    memcpy( u_sol   , sol , n*sizeof(float) ) ; 
    //for( int i = 0 ; i < n*n ; i++)
    //    u_input[i] = input[i] ;
    //for( int i = 0 ; i < n ; i++)
    //    u_sol[i] = sol[i] ;
    float* u_xk ; 
    float* u_xk1; 
    cudaMallocManaged(&u_xk , n*sizeof(float));
    cudaMallocManaged(&u_xk1, n*sizeof(float));

    //Kernel
    const int numBlock  = 160 ; 
    const int blocksize = 32 ; 
    clock_t c_start = clock();
    for ( int it = 0 ; it < iter ; it++){
        printf( "iter = %d \n" , it ) ;
        Unified_kernel<<< numBlock , blocksize  >>> ( n , numBlock , u_input , u_sol , u_xk , u_xk1 ) ;
        cudaDeviceSynchronize(); 
        swap_pointer( u_xk , u_xk1 ) ; 
    }
    clock_t c_end = clock();
    exeTime = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    
    printf( "Finish All iters   \n"  ) ;

    //Memback
    //for( int i = 0 ; i < n ; i++)
    //    x_k[i] = u_xk[i] ;
    
    memcpy( x_k   , u_xk , n*sizeof(float) ) ; 
    printf( "Finish Copy   \n"  ) ;
    
    //cudaFree(u_input);
    //cudaFree(u_sol);
    //cudaFree(u_xk);
    //cudaFree(u_xk1);
    //delete[] u_input; 
    //delete[] u_sol; 
    //delete[] u_xk; 
    //delete[] u_xk1; 
    printf( "Finish Func   \n"  ) ;
}



