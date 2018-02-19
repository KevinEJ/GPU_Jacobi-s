
using namespace std ;
// Parse input data 
void getinput( string filename , int* n , int* iter , float**& input, float*& sol ){

    string line;
    ifstream myfile( filename );
    if (myfile){  
        getline( myfile, line );   // First line  //printf( " [1] %s \n" , line.c_str() ) ;
        *n = stoi(line); 
        getline( myfile, line );   // Second line //printf( " [1] %s \n" , line.c_str() ) ;
        *iter = stoi(line) ; 

        input = new float* [*n] ;
        for( int i = 0 ; i < *n ; i++ )
            input[i] = new float[*n] ;
        sol = new float[*n] ;  

        //input = vector< vector<float> > ( *n , vector<float>( *n , 0 ) ) ;
        //sol = vector<float>( *n , 0 )  ;

        for( int i = 0 ; i < *n ; i++ ){
            getline( myfile, line) ; // get input line  // printf( " [3] %s \n" , line.c_str() ) ;
            istringstream iss(line) ;
            string s; 
            for( int j = 0 ; j < *n ; j++ ){ 
                getline( iss, s, ' ' ) ; // get input data   //printf( " [4] %s \n" , s.c_str() ) ;
                input[i][j] = stoi(s) ; 
            }
        }
        getline( myfile, line) ; // Solution  
        for( int i = 0 ; i < (*n) ; i++ ){
            getline( myfile, line) ;  //Get sol
            sol[i] = stoi(line) ;  
        }
        myfile.close();
    }
    else cout << "fooey\n";
}


// 2D Matric Multiply Funtion 
float* MatrixMultiple( float** A , float* x , int n){
    float* res = new float[n] ;
    for( int i = 0 ; i < n ; i++){
        float bi = 0 ; 
        for( int j = 0 ; j < n ; j++){
            bi += A[i][j] * x[j] ; 
        }
        res[i] = bi ; 
    }
    return res ;
}
