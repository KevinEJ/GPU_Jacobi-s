
void CPU_Jacobi( int n , int iter , float* input , float* sol , float* x_k , float* x_k1 ){
    
    for ( int it = 0 ; it < iter ; it++ ){
        printf( "iter = %d \n" , it ) ;
        for ( int i = 0 ; i < n ; i ++ ){
            float t = 0 ; 
            for( int j = 0 ; j < n ; j ++){
                if ( i!=j ){
                    t += input[i*n+j]*x_k[j] ; 
                }
            }
            x_k1[i] = ( sol[i] - t ) / input[i*n+i] ; 
        }
        swap_pointer( x_k , x_k1 ) ; 
    }

}
