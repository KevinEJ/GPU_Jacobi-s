

__global__ void 
J_kernel ( int n , float* input , float* sol , float* x_k , float* x_k1){
    
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
    //x_k1[idx] =  (input[idx][idx]) ;
    //float aaa = sol[idx] ;
    //x_k1[idx] = aaa   ;  
    
    // swap x_k and x_k1

    //x_k1[idx] = 3.0 ; // = sol[idx] ;
    //x_k[idx]  = 4.0 ; //= sol[idx] ;

    return ; 
}
