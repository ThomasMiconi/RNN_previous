//
//  The X method cannot work with the fixed-modif comparison!
//
//  Let's try to get any method (e.g. dxthistrial) working for seq-XOR with the simple error trace first  
//
//  OK This version seems to work with DXTRIAL method and the simple error
//  trace, though with much instability! The relevant variables are ETA,
//  ALPHABIAS and MAXDW.
//  Basically the main problem with simple error trace is to avoid too much
//  noise accumulation in the weights between successive evaluation

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <string>
#include <random>
#include "Eigen/Dense"

#define TESTING 777
#define LEARNING 666


#define NBPATTERNS  4

using namespace std;
using namespace Eigen;
void saveWeights(MatrixXd& m, string fname);
void readWeights(MatrixXd& m, string fname);
void randJ(MatrixXd& m);
void randVec(VectorXd & m);
void randMat(MatrixXd& m);


int NBNEUR = 200;
int NBIN = 5;  // Input 0 is reserved for a 'go' signal
int NBOUT = 1;
double PROBACONN = 1.0;
double G = 1.5;
string METHOD = "DXTRIAL";
double MAXDW = 5e-5 ; //* 1.5;
int RNGSEED = 20;
double PROBAMODUL = .0;
double ALPHAMODUL = .0;

double PROBANOISE = .0;
double ALPHANOISE = .0;

double PROBAHEBB = .1;
double ALPHABIAS = .01;

double ALPHATRACE = .6;

int SUBW=0;

double ETA =  .01 ; // * 1.5;  // Learning rate
double STIMVAL = .5;


std::default_random_engine myrng;
std::normal_distribution<double> Gauss(0.0,1.0);
std::uniform_real_distribution<double> Uniform(0.0,1.0);

int main(int argc, char* argv[])
{

    fstream myfile;


    int PHASE=LEARNING;
    if (argc > 1)
       for (int nn=1; nn < argc; nn++)
       {
           if (strcmp(argv[nn], "test") == 0) { PHASE = TESTING; cout << "Test mode. " << endl; }
           if (strcmp(argv[nn], "METHOD") == 0) { METHOD = argv[nn+1]; }
           if (strcmp(argv[nn], "SUBW") == 0) { SUBW = atoi(argv[nn+1]); }
           if (strcmp(argv[nn], "G") == 0) { G = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "ALPHABIAS") == 0) { ALPHABIAS = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "ETA") == 0) { ETA = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "STIMVAL") == 0) { STIMVAL = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "ALPHAMODUL") == 0) { ALPHAMODUL = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "NBNEUR") == 0) { NBNEUR = atoi(argv[nn+1]); }
           if (strcmp(argv[nn], "PROBAMODUL") == 0) { PROBAMODUL = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "PROBAHEBB") == 0) { PROBAHEBB = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "ALPHATRACE") == 0) { ALPHATRACE = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "RNGSEED") == 0) { RNGSEED = atof(argv[nn+1]); }
           if (strcmp(argv[nn], "MAXDW") == 0) { MAXDW = atof(argv[nn+1]); }
       }

    string SUFFIX = "_G" + to_string(G) + "_MAXDW" + to_string(MAXDW) + "_ETA" + to_string(ETA) + "_ALPHAMODUL" + to_string(ALPHAMODUL) + "_PROBAMODUL" + to_string(PROBAMODUL) + "_SUBW" +to_string(SUBW) + "_ALPHATRACE" + to_string(ALPHATRACE) + "_METHOD-" + METHOD + "_ALPHABIAS" + to_string(ALPHABIAS) + "_PROBAHEBB" + to_string(PROBAHEBB) + "_NBNEUR" + to_string(NBNEUR) + "_RNGSEED" + to_string(RNGSEED);
    cout << SUFFIX << endl;

    myrng.seed(RNGSEED);

    double dt = 1.0;
    double tau = 10.0;
    int trialtype;

    int NBTRIALS = 100017; 
    int TRIALTIME = 700 ; //500; // 1100
    int STARTSTIM1 = 1, TIMESTIM1 = 500; // 200
    //int STARTSTIM2 = 400, TIMESTIM2 = 0; 

    if (PHASE == TESTING) 
        NBTRIALS = 40*NBPATTERNS;


    MatrixXd patterns[NBPATTERNS];
    MatrixXd tgtresps[NBPATTERNS];


    // Remember that input channel 0 is reserved for the unimplemented 'go' signal
    

    MatrixXd dJ(NBOUT, NBNEUR); dJ.setZero();
    MatrixXd win(NBNEUR, NBIN); randMat(win); //win.setRandom();// win.row(0).setZero(); // Uniformly between -1 and 1, except possibly for output cell (not even necessary).
    cout << win.col(0).head(5) << endl;
    MatrixXd J(NBNEUR, NBNEUR);
    
    randJ(J);
    
    if (PHASE == TESTING){
        readWeights(J, "J.dat");
        readWeights(win, "win.dat");
    }



    VectorXd meanerrs(NBTRIALS); meanerrs.setZero();
    VectorXd lateral_input;
    VectorXd dxthistrial(NBNEUR);
    MatrixXd rs(NBNEUR, TRIALTIME); rs.setZero();
    MatrixXd hebb(NBNEUR, NBNEUR);  
    VectorXd x(NBNEUR), r(NBNEUR), rprev(NBNEUR), dxdt(NBNEUR), k(NBNEUR), 
             input(NBIN), deltax(NBNEUR); 
    x.fill(0); r.fill(0);
    
    VectorXd err(TRIALTIME); 
    VectorXd meanerrtrace(NBPATTERNS);
    double meanerr;

    MatrixXd dJtmp, Jprev, Jr;

    double dtdivtau = dt / tau;



    meanerrtrace.setZero();



    for (int numtrial=0; numtrial < NBTRIALS; numtrial++)
    {

        if (PHASE == LEARNING)
            //trialtype = (int)(numtrial/2) % NBPATTERNS;
            trialtype = numtrial % NBPATTERNS;
        else 
            trialtype = numtrial % NBPATTERNS;

        
        hebb.setZero();
        dJ.setZero();
        //input = patterns.col(trialtype);
        input.setZero();
        
        x.fill(0.0); 
        //x.setRandom(); x *= .05; 
        x(1)=1.0; x(10)=1.0; x(11)=-1.0; //x(12) = 1.0; 
        x += dtdivtau * win * input;
        for (int nn=0; nn < NBNEUR; nn++)
            r(nn) = tanh(x(nn));

        

        randVec(dxthistrial);
        dxthistrial *= ALPHABIAS;

        double tgtresp;
        double biasmodality1, biasmodality2;

        if (trialtype == 0){
            input(3) = 1.0; input(4) = 0.0;
            biasmodality1 = 1.0; 
            tgtresp = .98;
            biasmodality2 = Uniform(myrng) < .5 ?  1 : -1;
        }
        if (trialtype == 1){
            input(3) = 1.0; input(4) = 0.0;
            biasmodality1 = -1.0; 
            tgtresp = -.98;
            biasmodality2 = Uniform(myrng) < .5 ?  1 : -1;
        }
        if (trialtype == 2){
            input(3) = 0.0; input(4) = 1.0;
            biasmodality2 = 1.0; 
            tgtresp = .98;
            biasmodality1 = Uniform(myrng) < .5 ?  1 : -1;
        }
        if (trialtype == 3){
            input(3) = 0.0; input(4) = 1.0;
            biasmodality2 = -1.0; 
            tgtresp = -.98;
            biasmodality1 = Uniform(myrng) < .5 ?  1 : -1;
        }

        biasmodality1 *= .2;
        biasmodality2 *= .2;

        for (int numiter=0; numiter < TRIALTIME;  numiter++)
        {

            input(0) = 0;
            input(1) = 0; input(2) = 0;
            if (numiter >= STARTSTIM1  & numiter <  STARTSTIM1 + TIMESTIM1)
            {
                input(1) = .5 * Gauss(myrng) + biasmodality1;
                input(2) = .5 * Gauss(myrng) + biasmodality2;
            }
            rprev = r;
            lateral_input =  J * r;
        
            deltax = dtdivtau * ( -x + lateral_input /*+ wfb * zout */ + STIMVAL * win * input + dxthistrial );
            x += deltax;
        
            x(1)=1.0; x(10)=1.0;x(11)=-1.0; //x(12) = 1.0; 
            
            //if (numtrial % 2 == 1)
            { 
                if ((PHASE == LEARNING) && (Uniform(myrng) < PROBAHEBB))
                {
                    if (METHOD == "DXTRIAL")
                        hebb += r * dxthistrial.transpose();
                    else if (METHOD == "X")
                        hebb += r * x.transpose();
                    else if (METHOD == "DELTAX")
                        hebb += r * deltax.transpose();
                    else if (METHOD == "LATINPUT")
                        hebb += r * (dtdivtau * lateral_input.transpose()); // Works, with instabilities and sufficiently low ETA
                    else { cout << "Which method??" << endl; return -1; }
                }
            }

            for (int nn=0; nn < NBNEUR; nn++)
            {
                /*if (x(nn) > 0)
                    r(nn) = tanh(x(nn));
                else
                    r(nn) = .1 * tanh(10.0*x(nn));*/
                r(nn) = tanh(x(nn));
            }
           

            rs.col(numiter) = r;

            
        }
       
       int EVALTIME = 300; 

        err = rs.row(0).array() - tgtresp;
        err.head(TRIALTIME - EVALTIME).setZero();

        //meanerr = 2.0   * err.cwiseAbs().sum();
        meanerr =  err.cwiseAbs().sum() / (double)EVALTIME;

        if ((PHASE == LEARNING) && (numtrial> 100)
                // && (numtrial %2 == 1)
           )
            {
             
                
                // dJ = -.0001 * meanerr.sum() * (hebb.array() * (meanerr.sum() - meanerrtrace.col(trialtype).sum())).transpose().cwiseMin(5e-4).cwiseMax(-5e-4); << Version that worked
                //dJ = (  -.0000001 * meanerr * (hebb.array() * (meanerr - meanerrtrace(trialtype)))).transpose().cwiseMin(1e-6).cwiseMax(-1e-6); 
                //dJ = (  -.000005 * meanerr * (hebb.array() * (meanerr - meanerrtrace(trialtype)))).transpose().cwiseMin(5e-5).cwiseMax(-5e-5);
                //dJ = G * (  -  ETA * meanerr * (hebb.array() * (meanerr - meanerrtrace(trialtype)))).transpose().cwiseMin(MAXDW).cwiseMax(-MAXDW);
                
                //dJ = (  -  ETA * meanerrtrace(trialtype) * meanerrtrace(trialtype) * (hebb.array() * (meanerr - meanerrtrace(trialtype)))).transpose().cwiseMin(MAXDW).cwiseMax(-MAXDW);
                dJ = (  -  ETA * meanerrtrace(trialtype) * (hebb.array() * (meanerr - meanerrtrace(trialtype)))).transpose().cwiseMin(MAXDW).cwiseMax(-MAXDW);

                //J /= G;
                J +=  dJ;
                //J *= G;


            }


        meanerrtrace(trialtype) = ALPHATRACE * meanerrtrace(trialtype) + (1.0 - ALPHATRACE) * meanerr; 
        //meanerrtrace(trialtype) = meanerr; 
        meanerrs(numtrial) = meanerr;
       

        if (PHASE == LEARNING)
        {
            if (numtrial % 3000 < 8) 
            {
                //myfile.open("rs"+std::to_string((numtrial/2)%4)+".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();
                //myfile.open("rs"+std::to_string(trialtype)+".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();
                
                //myfile.open("rs"+std::to_string(numtrial % 3000)+".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();
            }
            if (numtrial % 3000 == 0)
            {
                //myfile.open("J_"+std::to_string(numtrial)+".txt", ios::trunc | ios::out);  myfile << J << endl; myfile.close();
                //saveWeights(J, "J_"+std::to_string(numtrial)+".dat");
                myfile.open("J" + SUFFIX + ".txt", ios::trunc | ios::out);  myfile << J << endl; myfile.close();
                myfile.open("win" + SUFFIX + ".txt", ios::trunc | ios::out);  myfile << win << endl; myfile.close();
                saveWeights(J, "J" + SUFFIX + ".dat");
                saveWeights(win, "win" + SUFFIX + ".dat"); // win doesn't change over time.
                
                myfile.open("errs" + SUFFIX + ".txt", ios::trunc | ios::out);  myfile << endl << meanerrs.head(numtrial) << endl; myfile.close();

            }


            if (numtrial % (NBPATTERNS * 200) <  2*NBPATTERNS)
            {    
                cout << numtrial << "- trial type: " << trialtype;
                //cout << ", responses : " << zout;
                //cout << ", time-avg responses for each pattern: " << zouttrace ;
                //cout << ", sub(abs(wout)): "  << wout.cwiseAbs().sum() ;
                //cout << ", hebb(0,1:3): " << hebb.col(0).head(4).transpose();
                cout << ", meanerr: " << meanerr;
                //cout << ", wout(0,1:3): " << wout.row(0).head(5) ; 
                cout << ", r(0,1:6): " << r.transpose().head(6) ; 
                cout << ", dJ(0,1:4): " << dJ.row(0).head(4)  ;
                cout << endl;
            }
        }
        else if (PHASE == TESTING) {
            cout << numtrial << "- trial type: " << trialtype;
            cout << " r[0]: " << r(0);
            cout << endl;
            myfile.open("rs_long_type"+std::to_string(trialtype)+"_"+std::to_string(int(numtrial/NBPATTERNS))+".txt", ios::trunc | ios::out);  myfile << endl << rs.transpose() << endl; myfile.close();
        }


    }

    cout << "Done learning ..." << endl;


    cout << J.mean() << " " << J.cwiseAbs().sum() << " " << J.maxCoeff() << endl;
    //cout << wout.mean() << " " << wout.cwiseAbs().sum() << " " << wout.maxCoeff() << endl;


    cout << endl;
    return 0;
}




void saveWeights(MatrixXd& J, string fname)
{
    double wdata[J.rows() * J.cols()];
    int idx=0;
    for (int cc=0; cc < J.cols(); cc++)
        for (int rr=0; rr < J.rows(); rr++)
            wdata[idx++] = J(rr, cc);
    ofstream myfile(fname, ios::binary | ios::trunc);
    if (!myfile.write((char*) wdata, J.rows() * J.cols() * sizeof(double)))
        throw std::runtime_error("Error while saving matrix of weights.\n");
    myfile.close();
}
void readWeights(MatrixXd& J, string fname)
{
    double wdata[J.cols() * J.rows()];
    int idx=0;
    cout << endl << "Reading weights from file " << fname << endl;
    ifstream myfile(fname, ios::binary);
    if (!myfile.read((char*) wdata, J.cols() * J.rows() * sizeof(double)))
        throw std::runtime_error("Error while reading matrix of weights.\n");
    myfile.close();
    for (int cc=0; cc < J.cols() ; cc++)
        for (int rr=0; rr < J.rows(); rr++)
            J(rr, cc) = wdata[idx++];
    cout << "Done!" <<endl;
}

void randVec(VectorXd& M)
{
    for (int nn = 0; nn < M.size(); nn++)
        M.data()[nn] = -1.0 + 2.0 * Uniform(myrng);
}
void randMat(MatrixXd& M)
{
    for (int nn = 0; nn < M.size(); nn++)
        M.data()[nn] = -1.0 + 2.0 * Uniform(myrng);
}

void randJ(MatrixXd& J)
{
    for (int rr=0; rr < J.rows(); rr++)
        for (int cc=0; cc < J.cols(); cc++)
        {
            if (Uniform(myrng) < PROBACONN)
                J(rr, cc) =  G * Gauss(myrng) / sqrt(PROBACONN * NBNEUR);
            else
                J(rr, cc) = 0.0;
        }
}

/*
load rs0.txt; load rs1.txt; load rs2.txt; load rs3.txt;
figure; plot(rs0(1:7,:)'); figure; plot(rs1(1:7,:)'); figure; plot(rs2(1:7,:)'); figure; plot(rs3(1:7,:)'); 

r=load('resp0.txt');
figure; plot(r(2:8:end)); hold on; plot(r(3:8:end), 'r'); plot(r(5:8:end), 'g'); plot(r(7:8:end), 'm'); hold off

*/
