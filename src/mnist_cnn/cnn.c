#include <math.h>
#include <stdint-gcc.h>
#include "mnist_params.h"
#include "mnist_data.h"
#include  "demo_system.h"

#define RFW			6
#define L1W			12
#define L2W			4
#define NN_LAYER1	12
#define NN_LAYER2	32
#define NN_OUTPUT	10

int32_t cno1[NN_LAYER1][L1W][L1W];
int32_t cno2[NN_LAYER2][L2W][L2W];
int32_t l3[NN_OUTPUT];

//-------------------------------------------------
int cnn_Recogn(int testIndex, uint32_t *compute_cycles)
{
    *compute_cycles = 0;

    uint8_t iret=0;
    int32_t fsum,z;
    int32_t sq[NN_OUTPUT];
    int32_t se[NN_OUTPUT];
    uint8_t n,nn,cx,cy,rx,ry;
    int32_t qmax = 100000000;

    for(n=0;n<NN_LAYER1;n++) for(cy=0;cy<L1W;cy++) for(cx=0;cx<L1W;cx++)
    {
        fsum=thr1[n];
        for(ry=0;ry<RFW;ry++) for(rx=0;rx<RFW;rx++)
            {
                fsum+=(cw1[n][ry][rx]*mnist_test_data[testIndex][2*cy+ry][2*cx+rx]) >> 8;
            }
        if(fsum>0) z = fsum;
        else	   z = 0;
        cno1[n][cy][cx] = z;
    }

    for(n=0;n<NN_LAYER2;n++) for(cy=0;cy<L2W;cy++) for(cx=0;cx<L2W;cx++)
    {
        fsum=thr2[n];
        for(nn=0;nn<NN_LAYER1;nn++) for(ry=0;ry<RFW;ry++) for(rx=0;rx<RFW;rx++)
                {
                    fsum+=cw2[n][nn][ry][rx]*cno1[nn][2*cy+ry][2*cx+rx];
                }
        if(fsum>0) z = fsum;
        else	   z = 0;
        cno2[n][cy][cx] = z;
    }

    for(n=0;n<NN_OUTPUT;n++)
    {
        fsum=thro[n];
        for(nn=0;nn<NN_LAYER2;nn++) for(cy=0;cy<L2W;cy++) for(cx=0;cx<L2W;cx++)
                {
                    fsum+=w3[n][nn][cy][cx]*cno2[nn][cy][cx];
                }
        sq[n]=fsum;
    }

    z=-qmax;
    for(n=0;n<NN_OUTPUT;n++)
    {
        if(z<sq[n])
        {
            z=sq[n];
        }
    }

    for(n=0;n<NN_OUTPUT;n++) sq[n]-=z;
    for(n=0;n<NN_OUTPUT;n++) se[n]=exp(sq[n]);

    fsum=0;
    for(n=0;n<NN_OUTPUT;n++) fsum+=se[n];
    for(n=0;n<NN_OUTPUT;n++) l3[n]=se[n]/fsum;

    z=-1.0;
    for(n=0;n<NN_OUTPUT;n++)
    {
        if(l3[n]>z)
        {
            z=l3[n];
            iret=n;
        }
    }

    *compute_cycles += get_mcycle();

    return iret;
}