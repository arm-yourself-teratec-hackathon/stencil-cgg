#include <iostream>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
double
dml_micros()
{
        static struct timezone tz;
        static struct timeval  tv;
        gettimeofday(&tv,&tz);
        return((tv.tv_sec*1000000.0)+tv.tv_usec);
}
using namespace std;

typedef unsigned long long ui64;

const ui64 order=8;
ui64 DIMX,DIMY,DIMZ,iters;
ui64 MAXX,MAXY,MAXZ;
ui64 xyplane,MATsize;

// retourne un offset dans le centre de la matrice les dimensions sont [0..DIM-1]
inline
ui64 DIMXYZ(ui64 x,ui64 y,ui64 z){
        return((z+order)*xyplane+(y+order)*MAXX+x+order);
}

// retourne un offset dans la matrice les dimensions sont [-order..DIM+order-1] mais en indices de [0..DIM+2*order-1]
inline
ui64 MATXYZ(ui64 x,ui64 y,ui64 z){
        return(x+ y*MAXX+z*xyplane);
}

double *matA;
double *matB;
double *matC;


void init()
{

        // l initialisation ne fait pas partie de l exercise , elle peut etre optimisee mais n est pas mesuree car elle remplie de facon artificielle les matrices
        // les donnees n influent pas sur la performance

        // dynamically allocate memory of size DIMX*DIMY*DIMZ+ghost region on 6 faces
        matA = new double[MATsize];
        assert( matA!=NULL);
        matB = new double[MATsize];
        assert( matB!=NULL);
        matC = new double[MATsize];
        assert( matC!=NULL);

        // Initialisation centre et bords
        // Les matrices A et C sont mises a zero
        // A en la matrice d emtree et C la matrice de sortie
        // La matrice B est un stencil constant pour le run
        for (ui64 z = 0; z < MAXZ; z++) {
                for (ui64 y = 0; y < MAXY; y++){
                        for (ui64 x = 0; x < MAXX; x++){
                                matA[MATXYZ(x,y,z)] = 0.0;
                                matC[MATXYZ(x,y,z)] = 0.0;
                                matB[MATXYZ(x,y,z)] = sin(z*cos(x+0.311)*cos(y+.817)+.613);
                        }
                }
        }
        // Initialisation centre de A qui est la matrice de data
        for (ui64 z = 0; z < DIMZ; z++) {
                for (ui64 y = 0; y < DIMY; y++){
                        for (ui64 x = 0; x < DIMX; x++){
                                matA[DIMXYZ(x,y,z)] = 1.0;
                        }
                }
        }

}

void one_iteration()
{
                for (ui64 z = 0; z < DIMZ; z++) {
                        for (ui64 y = 0; y < DIMY; y++){
                                for (ui64 x = 0; x < DIMX; x++){
                                        matC[DIMXYZ(x,y,z)] = matA[DIMXYZ(x,y,z)]*matB[DIMXYZ(x,y,z)] ;
                                        for (ui64 o = 1; o <= order; o++){
                                               matC[DIMXYZ(x,y,z)]+= matA[DIMXYZ(x+o,y,z)]*matB[DIMXYZ(x+o,y,z)] / pow(17.0,o);
                                               matC[DIMXYZ(x,y,z)]+= matA[DIMXYZ(x-o,y,z)]*matB[DIMXYZ(x-o,y,z)] / pow(17.0,o);
                                               matC[DIMXYZ(x,y,z)]+= matA[DIMXYZ(x,y+o,z)]*matB[DIMXYZ(x,y+o,z)] / pow(17.0,o);
                                               matC[DIMXYZ(x,y,z)]+= matA[DIMXYZ(x,y-o,z)]*matB[DIMXYZ(x,y-o,z)] / pow(17.0,o);
                                               matC[DIMXYZ(x,y,z)]+= matA[DIMXYZ(x,y,z+o)]*matB[DIMXYZ(x,y,z+o)] / pow(17.0,o);
                                               matC[DIMXYZ(x,y,z)]+= matA[DIMXYZ(x,y,z-o)]*matB[DIMXYZ(x,y,z-o)] / pow(17.0,o);
                                        }
                                }
                        }
                }
                //  A=C
                for (ui64 z = 0; z < DIMZ; z++) {
                        for (ui64 y = 0; y < DIMY; y++){
                                for (ui64 x = 0; x < DIMX; x++){
                                        matA[DIMXYZ(x,y,z)] = matC[DIMXYZ(x,y,z)];
                                }
                        }
                }
}

int main(const int argc,char **argv)
{
        try {
                DIMX  = std::stoi(argv[1]);
                DIMY  = std::stoi(argv[2]);
                DIMZ  = std::stoi(argv[3]);
                iters = std::stoi(argv[4]);
                MAXX=DIMX+2*order;
                MAXY=DIMY+2*order;
                MAXZ=DIMZ+2*order;
                xyplane=MAXX*MAXY;
                MATsize=MAXX*MAXY*MAXZ;
        }

        catch (...) {
                cout << argv[0] << " siseX sizeY sizeZ iters" << endl;
                return -1;
        }

	init();

        //phase1
        for (ui64 i = 0; i < iters; i++) {
                // calcule 1 iteration Jacobi   C=B@A
                double t1=dml_micros();
                one_iteration();
                double t2=dml_micros();
                printf("_0_ ");
                for(ui64 i=0;i<5;i++)printf(" %18.15lf",matA[DIMXYZ(DIMX/2+i,DIMY/2+i,DIMZ/2+i)]);
                double ns_point=(t2-t1)*1000.0/DIMX/DIMY/DIMZ;
                printf("  %10.0lf  %10.3lf %lld %lld %lld\n",t2-t1,ns_point,DIMX,DIMY,DIMZ);
        }

        delete[] matA;
        delete[] matB;
        delete[] matC;

        return 0;

}
