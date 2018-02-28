#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>

#include "util.h"
#include "J_kernel.h"

int main(int argc, char *argv[]){
    
    // Parsing input 
    if( argc != 2 )
        cerr << " Usage: ./J_CPU input_n \n" ;
    
    string input_num =  argv[1]  ; 
    string filename = "inputs/" + input_num + ".txt" ; 

    int* n , *iter ;
    float* input ;
    float* sol; 

    n = new int(0) ; 
    iter = new int(0) ; 
    getinput( filename ,n , iter , input , sol ) ; 
    printf( " n = %d \n" , *n ) ;
    
    /*for( int i = 0 ; i < *n ; i++){
        printf( " c[%d] = %f \n" , i , sol[i] ) ;
    }
    for( int i = 0 ; i < *n ; i++){
        for( int j = 0 ; j < *n ; j++)
            printf( "%f , " ,  input[ i*(*n) + j] ) ;
        printf( "\n") ;
    }*/
    
    float* x_k  =  new float[*n]; 
    //float* x_k1 =  new float[*n]; 
    float* d_x_k  ; //= new float[*n]; 
    float* d_x_k1 ; //= new float[*n]; 
    
    
    for( int i = 0 ; i < *n ; i++){
        x_k[i] = 0 ;
    }
    cudaMalloc((void**)&d_x_k , (*n)*sizeof(float)) ; 
    cudaMalloc((void**)&d_x_k1, (*n)*sizeof(float)) ; 
    cudaMemcpy( d_x_k , x_k , (*n)*sizeof(float) , cudaMemcpyHostToDevice);
    //cudaMemcpy( d_x_k1, &x_k1, (*n)*sizeof(float) , cudaMemcpyHostToDevice);
    float *d_input, *d_sol ; 
    cudaMalloc((void**)&d_input  , (*n)*(*n)*sizeof(float)) ; 
    cudaMalloc((void**)&d_sol    , (*n)*sizeof(float)) ; 
    cudaMemcpy( d_input , input , (*n)*(*n)*sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy( d_sol   , sol   , (*n)*sizeof(float) , cudaMemcpyHostToDevice);

    clock_t c_start = clock();
    //for ( int it = 0 ; it < *iter ; it++){
    cudaStream_t streams[1024] ; 
    
    for( int i  = 0 ; i < 1024 ; i++){
        cudaStreamCreate( &streams[i] );        
    }
    
    for ( int it = 0 ; it < *iter ; it++){
        printf( "iter = %d \n" , it ) ;
        //J_kernel<<< 32 , 32  >>> ( *n , d_input , d_sol , d_x_k , d_x_k1 ) ;
        J_kernel<<< 80 , 64  >>> ( *n , 320 , d_input , d_sol , d_x_k , d_x_k1 ) ;
        //J_kernel<<< 64 , 80  >>> ( *n , 64 , d_input , d_sol , d_x_k , d_x_k1 ) ;
        cudaDeviceSynchronize(); 
        float* temp ; 
        temp = d_x_k ; 
        d_x_k = d_x_k1 ;
        d_x_k1 = temp ; 
    }
    cudaDeviceSynchronize(); 
    printf( "finish Jacobi \n") ;
/*    for ( int it = 0 ; it < *iter ; it++ ){
        printf( "iter = %d \n" , it ) ;
        for ( int i = 0 ; i < *n ; i ++ ){
            float t = 0 ; 
            for( int j = 0 ; j < *n ; j ++){
                if ( i!=j ){
                    t += input[i][j]*x_k[j] ; 
                }
            }
            x_k1[i] = ( sol[i] - t ) / input[i][i] ; 
        }
        float* temp ; 
        temp = x_k ; 
        x_k = x_k1 ;
        x_k1 = temp ; 
    }
*/  
    clock_t c_end = clock();

    cudaMemcpy( x_k   , d_x_k   , (*n)*sizeof(float) , cudaMemcpyDeviceToHost);
    
    for( int i = 0 ; i < *n ; i++){
        printf( " x[%d] = %f \n" , i , x_k[i] ) ;
    }

    float* res = MatrixMultiple( input , x_k , *n) ;  
    for( int i = 0 ; i < *n ; i++){
        printf( " res[%d] = %f    |  sol[%d] = %f  \n" , i , res[i] , i , sol[i] ) ;
    }

    double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    cout << "CPU time used: " << time_elapsed_ms/1000.0 << " ms\n";

    cudaFree(d_input); 
    cudaFree(d_sol); 
    cudaFree(d_x_k); 
    cudaFree(d_x_k1);
    delete[] input;
    delete[] sol;
    delete[] x_k;
    delete n ;
    delete iter ;

    return 0 ;
}

/*
GPU 
Naive : 
    for( k iteratinos )
        kernel<<<>>>
            for( all a[thread.x][j] ) 
                t += a * x_old
            x_new[thread.x] = ( b[ thread.x ] - t ) / a[i][i] ; 
Shared Memory 

*/


