
#include <iostream>
#include <string>
#include <ctime>

#include "util.h"
#include "CPU_Jacobi.h"
#include "GPU_Unified.h"
#include "GPU_MemCopy.h"
#include "GPU_Shared.h"
//#include "GPU_Stream.h"
#include "GPU_MemCoa.h"
#include "GPU_Reduce.h"
#include "GPU_Reduce2.h"
#include "GPU_Reduce3.h"


double exeTime ; 
int    g_Block_size ; 

int main(int argc, char *argv[]){

    // Parsing input 
    if( argc != 3 )
        cerr << " Usage: ./Jacobi input_n mode\n" ;
    
    int mode = stoi(argv[2]) ; // 0: CPU , 1: Unified, 2: Memcopy, 3:Shared, 4:....
    //g_Block_size = stoi(argv[3]) ; // 0: CPU , 1: Unified, 2: Memcopy, 3:Shared, 4:....
    string input_num =  argv[1]  ; 
    string filename = "inputs/" + input_num + ".txt" ; 

    //Declare General Variables
    int n , iter ;
    float* input ;
    float *sol , *x_k , *x_k1 ; 

    getinput( filename , n , iter , input , sol ) ; 
    printf( " n = %d \n" , n ) ;

    x_k  = new float[n] ; 
    x_k1 = new float[n] ; 
  
    for( int i = 0 ; i < n ; i++)
        x_k[i] = 0 ;
    
    // Tans input 
    float* t_input = new float[ n*n ] ; 
    for( int i = 0 ; i < n ; i++){
        for( int j = 0 ; j < n ; j ++ ){
            t_input[ j*n + i ]  = input[ i*n + j ] ; 
        }
    }

    //Implements 
    clock_t c_start = clock();
    //Mem copy 
    //clock_t c_mem_start = clock();
    if( mode == 0 )
        CPU_Jacobi( n , iter , input , sol , x_k , x_k1 ) ; 
    else if( mode == 1 )
        GPU_Unified( n , iter , input , sol , x_k , x_k1 ) ; 
    else if( mode == 2 )
        GPU_MemCopy( n , iter , input , sol , x_k , x_k1 ) ; 
    else if( mode == 3 )
        GPU_Shared( n , iter , t_input , sol , x_k , x_k1 ) ; 
    else if( mode == 4 )
        GPU_Memcoalesc( n , iter , t_input , sol , x_k , x_k1 ) ; 
    else if( mode == 5 )
        GPU_Reduction( n , iter , input , sol , x_k , x_k1 ) ; 
    else if( mode == 6 )
        GPU_Reduction2( n , iter , input , sol , x_k , x_k1 ) ; 
    else if( mode == 7 )
        GPU_Reduction3( n , iter , input , sol , x_k , x_k1 ) ; 
    //clock_t c_mem_end = clock();

    //Kernel Call
    //clock_t c_exe_start = clock();
    
    //clock_t c_exe_end = clock();

    //Mem copy back 
    //clock_t c_memback_start = clock();
    //clock_t c_memback_end = clock();
    clock_t c_end = clock();

    //Verification
    float* res = MatrixMultiple( input , x_k , n) ; 
    bool check = true ; 
    //print_1D_array( n , "x" , x_k ) ;
    for( int i = 0 ; i < n ; i++){
        if( abs(res[i]-sol[i]) > 1){
            printf( "Answer is wrong !! \n" );
            check = false;
        }
    //    printf( " res[%d] = %f    |  sol[%d] = %f  \n" , i , res[i] , i , sol[i] ) ;
    }
    if( check ){
        printf( "Answer is correct \n");
    }

    cudaDeviceSynchronize();
    //Delete 
    delete[] input ; 
    delete[] sol ; 
    delete[] x_k ; 
    delete[] x_k1 ; 
    delete[] res ; 
    /*  
    double memcopy_time = 1000.0 * (c_mem_end-c_mem_start) / CLOCKS_PER_SEC;
    cout << "Memcopy time used: " << memback_time/1000.0 << " s\n";
    double exe_time     = 1000.0 * (c_exe_end - c_exe_start) / CLOCKS_PER_SEC;
    cout << "Execute time used: " << exe_time    /1000.0 << " s\n";
    double memback_time = 1000.0 * (c_memback_end-c_memback_start) / CLOCKS_PER_SEC;
    cout << "Memback time used: " << memback_time/1000.0 << " s\n";
    */
    double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    cout << "Total  time used: " << time_elapsed_ms/1000.0 << " s\n";
    cout << "Kernel time used: " << exeTime/1000.0 << " s\n";
    return 0; 
}
