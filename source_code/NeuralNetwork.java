//Author: God Bennett , 2023, march 15
//Title: Neural network from scratch/memory 

public class NeuralNetwork 
{
    //features
    private double eta = 0.2;
    private double alpha = 0.5;
    private Architecture architecture = new Architecture ( "2,2,1" );
    private Layers layers = new Layers ( );
    
    
    //constructor
    public NeuralNetwork ( )
    {
        for ( int lSI = 0; lSI < architecture.size ( ); lSI ++ )
        {
            layers.add ( new Layer ( ) );
            
            for ( int lI = 0; lI <= architecture.get ( lSI ); lI ++ ) //<= implies bias neuron
            {
                int numberOfWeightsFromNextNeuron = ( lSI + 1 < architecture.size ( ) ? architecture.get ( lSI + 1 ) : 0 );
                
                Neuron newNeuron = new Neuron ( eta, alpha, lI, numberOfWeightsFromNextNeuron );
                
                layers.get ( lSI ).add ( newNeuron );
                layers.get ( lSI ).get ( layers.get ( lSI ).size ( ) - 1 ).setOutcome ( 1.0 ); //bias neuron last neuron per layer
            }
        }
    }
    
    
    //methods
    public void doForwardPropagation ( int [ ] inputs )
    {
        //set layer 0 outcomes to each input space
        for ( int iI = 0; iI < inputs.length; iI ++ )
            layers.get ( 0 ).get ( iI ).setOutcome ( inputs [ iI ] );
            
        //layers iteration index lSI
        for ( int lSI = 1; lSI < architecture.size ( ); lSI ++ )
        {
           Layer priorLayer = layers.get ( lSI - 1 ); 
            
           //layers iteration index lSI
           for ( int lI = 0; lI < architecture.get ( lSI ); lI ++ ) 
           {
               layers.get ( lSI ).get ( lI ).doForwardPropagation ( priorLayer );
           }
        }
    }
    
    public void doBackwardPropagation ( int target )
    {
        //calc outcome gradient
        Neuron outcomeNeuron = layers.get ( layers.size ( ) - 1 ).get ( 0 );
        outcomeNeuron.calculateOutcomeGradient ( target );
        
        //calc hidden
        for ( int lSI = architecture.size ( ) - 2; lSI > 0; lSI -- )
        {
            Layer currentLayer = layers.get ( lSI );
            Layer nextLayer = layers.get ( lSI + 1 );
            
            for ( int lI = 0; lI < currentLayer.size ( ); lI ++ ) 
                currentLayer.get ( lI ).calculateHiddenGradient ( nextLayer );
        }
        
        //update weights
        for ( int lSI = architecture.size ( ) - 1; lSI > 0; lSI -- )
        {
            Layer currentLayer = layers.get ( lSI );
            Layer priorLayer = layers.get ( lSI - 1 );
            
            for ( int lI = 0; lI < currentLayer.size ( ) - 1; lI ++ ) 
                currentLayer.get ( lI ).updateWeights ( priorLayer );
        }
    }
    
    public double getOutcome ( )
    {
        return layers.get ( layers.size ( ) - 1 ).get ( 0 ).getOutcome ( );
    }
}