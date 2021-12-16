#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define INPUTS 23
#define OUTPUTS 1
#define NODES 26
#define LAYERS 2

#define STOP_CONDITION 9000000

#define FILENAME_R "best"
#define EXTENSION "txt"
#define LINE 300

static double desired[OUTPUTS],bias[LAYERS],array_error_out[2][NODES],learning_rate=0.0026459;
static int n_target_size[LAYERS+1],w_target_size[LAYERS],layer_activation[LAYERS]={1,0},l_target=LAYERS+1;


typedef struct node{
    double **weights;
    double output;
}nodes;

typedef struct layer{
    int activation_function;
    nodes *n;
}layers;

typedef struct network_structure{
    layers *structure;
}ANN;

void create_network(ANN *network);
void initalize_weights(nodes *p_node,int w_target);
void free_network(ANN *network);
void readData(ANN *network,char *oneline);
void feed_forward(ANN *network,int q);

double activation_Sigmoid(double sum);
double activation_Tanh(double sum);
double activation_ReLu(double sum);
double (*activation[3])(double sum)={activation_Sigmoid,activation_Tanh,activation_ReLu};
double back_propagation(ANN *network,int q, int p);

void accuracy_tracker(int n,double output);
void E_total(ANN *network);
double error_out(double desired, double output);
double partial_derivative_Tanh(double output);
double partial_derivative_Sigmoid(double output);
double partial_derivative_ReLu(double output);
double (*activation_derivative[3])(double output)={partial_derivative_Sigmoid,partial_derivative_Tanh, partial_derivative_ReLu};

int main()
{
    FILE *fp;
    static int l,q,p,lines,epoch;
    char *token, oneline[LINE],filename[40],buffer[40]={FILENAME_R};

    srand(time(NULL));

    ANN neuralnet;
    layers *p_layer;
    nodes *p_node;

    l_target=LAYERS+1;
    for(l=0;l<l_target;l++){
        n_target_size[l]=NODES;
    }
    n_target_size[0]=INPUTS;
    n_target_size[LAYERS]=OUTPUTS;

    for(l=0;l<l_target-1;l++){
        bias[l]=1;
    }

    create_network(&neuralnet);
    while(epoch<STOP_CONDITION){
        if(fp==EOF||fp==NULL){

                snprintf(filename, sizeof(char) * 32, "%s0.%s",buffer,EXTENSION);
                fp=fopen(filename,"r");
                if(fp==NULL){
                    printf("No file named %s was found\nEnd training session",filename);
                    break;
                }
                while(fgets(oneline, LINE-1, fp) != NULL){
                    lines++;
                }
                fclose(fp);
        }
        fp=fopen(filename,"r");

        for(l=0;l<lines;l++){
            epoch++;
            q=(epoch+1)%2;
            p=(epoch)%2;

            fgets(oneline,LINE-1,fp);

            readData(&neuralnet,oneline);
            feed_forward(&neuralnet,q);

            E_total(&neuralnet);
            back_propagation(&neuralnet,q,p);
        }
        fclose(fp);
    }

    free_network(&neuralnet);

return 0;
}

/* FUNCTIONS FOR ALLOCATING MEMORY FOR NETWORK, INITALIZING WEIGHTS & FREEING ALLOCATED MEMORY */

/* Creates a neural network with the specified parameters */
void create_network(ANN *network){
    int l,t,n,k,n_target,w_target;
    layers *p_layer;
    nodes *p_node;
    network->structure=malloc(l_target*sizeof(layers));
    p_layer=network->structure+0;
    p_layer->n=malloc(INPUTS*sizeof(nodes));


    for(l=1,t=0;l<l_target;l++,t++){
        n_target=n_target_size[l];
        p_layer=network->structure+l;
        w_target=n_target_size[t]+1;
        p_layer->n=malloc(n_target*sizeof(nodes));
        for(n=n_target;n--;){
            p_node=p_layer->n+n;
            p_node->weights=malloc(2*sizeof(double*));
            for(k=2;k--;){

                p_node->weights[k]=malloc(w_target*sizeof(double));
            }
            initalize_weights(p_node,w_target);
        }

    }

    for(l=0;l<LAYERS;l++){
        p_layer=network->structure+l+1;
        p_layer->activation_function=layer_activation[l];
    }

}
/* Initializes weights within the create_network() function */
void initalize_weights(nodes *p_node,int w_target){
    int w;

    for(w=w_target;w--;){
        p_node->weights[0][w]=((double)rand()/32767.0)*2-1;
    }
}

/* Frees allocated memory */
void free_network(ANN *network){
    int l,t,n,w,k,n_target,w_target;
    layers *p_layer;
    nodes *p_node;
    w_target=INPUTS;
    for(l=1,t=0;l<l_target;l++,t++){
            n_target=n_target_size[l];
            p_layer=network->structure+l;
            for(n=n_target;n--;){
                p_node=p_layer->n+n;
                for(k=2;k--;){
                        free(p_node->weights[k]);
                }
                free(p_node->weights);

            }
            w_target=NODES;
            free(p_layer->n);
        }
        p_layer=network->structure+0;
        free(p_layer->n);
        free(network->structure);
        printf("Freed allocated memory");
}
/* FEEDFORWARD FUNCTION AND ACTIVATION FUNCTIONS */
void readData(ANN *network,char *oneline){
    int n;
    char *token;
    layers *p_layer;
    nodes *p_node;

    p_layer=network->structure+0;
    token=strtok(oneline,"\t :;| []()");
    for(n=INPUTS;n--;){
        p_node=p_layer->n+n;
        p_node->output=atof(token);
        token=strtok(NULL,"\t :;| []()");
    }
    for(n=OUTPUTS;n--;){
        desired[n]=atof(token);
        token=strtok(NULL,"\t :;-| ,[]()");
    }
}
void feed_forward(ANN *network,int q){
    int l,t,w,n,n_target,w_target;
    layers *p_layer;
    layers *p2_layer;
    nodes *p_node;
    nodes *p2_node;


    for(l=1,t=0;l<l_target;l++,t++){
        n_target=n_target_size[l];
        w_target=n_target_size[t];
        p_layer=network->structure+l;
        p2_layer=network->structure+t;
        for(n=n_target;n--;){
            p_node=p_layer->n+n;
            p_node->output=0;
            for(w=w_target;w--;){
                p2_node=p2_layer->n+w;
                p_node->output+=p2_node->output*p_node->weights[q][w];
            }
            p_node->output+=bias[t]*p_node->weights[q][w+1];
            p_node->output=(*activation[p_layer->activation_function])(p_node->output);
        }
    }
}

double activation_Sigmoid(double sum){
    double output;
    output= 1.0/( 1.0+exp(-sum) );

    return output;
}
double activation_Tanh(double output){
    double Sinh,Cosh;
    Sinh = exp( output ) - exp( -output );
    Cosh = exp( output ) + exp( -output );

    return Sinh / Cosh;
}
double activation_ReLu(double output){

    return output>0?output:0;
}


/*BACKPROPAGATION FUNCTION ETOTAL AND DERIVATIVE FUNCTIONS */
double back_propagation(ANN *network,int q, int p){
    int l,t,n,w,d=0,b=0,n_target,w_target,w2_target;
    double error_out_derivative;
    layers *p_layer;
    layers *p2_layer;
    layers *p3_layer;
    nodes *p_node;
    nodes *p2_node;


     //This can be a function
    d=LAYERS%2;
    void OutputDelta(array_error_out);{
    void HiddenDelta();
    p_layer=network->structure+LAYERS;
    p2_layer=network->structure+LAYERS-1;
    n_target=OUTPUTS;
    w_target=NODES;

    for(n=n_target;n--;){
        p_node=p_layer->n+n;
        error_out_derivative=error_out(desired[n],p_node->output);
        array_error_out[d][n]=error_out_derivative*(*activation_derivative[p_layer->activation_function])(p_node->output);
        // this can be a function
        for(w=w_target;w--;){
            p2_node=p2_layer->n+w;
            p_node->weights[p][w]=p_node->weights[q][w]-learning_rate*array_error_out[d][n]*p2_node->output;
        }
        p_node->weights[p][n_target]=p_node->weights[p][n_target]-learning_rate*array_error_out[d][n]*bias[LAYERS-1];

    }
    error_out_derivative=0;
    }
     //This can be another function
    for(l=LAYERS,t=LAYERS-1;1<l--,t--;){
        d=(l+1)%2;
        b=(l)%2;
        n_target=n_target_size[l];
        w_target=n_target_size[l+1];
        w2_target=n_target_size[t];
        p_layer=network->structure+l;
        for(n=n_target;n--;){
            p_node=p_layer->n+n;
            for(w=w_target;w--;){
                p2_layer=network->structure+l+1;
                p2_node=p2_layer->n+w;
                error_out_derivative+=array_error_out[d][w]*p2_node->weights[q][n];
            }
            p2_layer=network->structure+t;
            array_error_out[b][n]=error_out_derivative*(*activation_derivative[p_layer->activation_function])(p_node->output);

             // this can be a function
            for(w=w2_target;w--;){
                p2_node=p2_layer->n+w;
                p_node->weights[p][w]=p_node->weights[q][w]-learning_rate*array_error_out[b][n]*p2_node->output;

            }

            p_node->weights[p][w2_target]=p_node->weights[p][w2_target]-learning_rate*array_error_out[b][n]*bias[t];
            error_out_derivative=0;
        }
    }
}
void accuracy_tracker(int n,double output){
        static int correct, wrong;
        if(desired[n]==0){
            correct+=output<0.5;
            wrong+=output>0.5;
        }
        else{
            correct+=output>0.5;
            wrong+=output<0.5;
        }
}
void E_total(ANN *network){
    double error;
    int n;
    layers *p_layer=network->structure+LAYERS;
    nodes *p_node;

    for(n=0;n<OUTPUTS;n++){
        p_node=p_layer->n+n;
        error+=pow(desired[n]-p_node->output,2)*0.5;
        accuracy_tracker(n,p_node->output);
        printf("t%d %-12.5lg | y%-12.5lg = %d  correct %d\n",n,desired[n],p_node->output,p_node->output>0.5, (int)desired[n]==(p_node->output>0.5));

    }
    error=error/OUTPUTS;
}

double error_out(double desired, double output){

    return -( desired-output );
}

double partial_derivative_Tanh(double output){

    return 1.0-output*output;
}

double partial_derivative_Sigmoid(double output){

    return output*( 1.0-output );
}
double partial_derivative_ReLu(double output){

    return output=output>0;;
}
