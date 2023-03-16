//Author: God Bennett , 2023, march 15
//Title: Neural network from scratch/memory
import java.util.ArrayList;
import java.util.Random;

public class Neuron
{
    //features
    private int numberOfWeightsFromNextNeuron;
    private int neuronId;
    private double outcome;
    private double gradient;
    private ArrayList <Synapse> weights;
    private double eta;
    private double alpha;
    
    //constructor
    public Neuron ( double eta, double alpha, int neuronId, int numberOfWeightsFromNextNeuron )
    {
        this.neuronId = neuronId;
        this.eta = eta;
        this.alpha = alpha;
        this.numberOfWeightsFromNextNeuron = numberOfWeightsFromNextNeuron;
        this.gradient = 0.0;
        
        weights = new ArrayList <Synapse> ( );
        
        for ( int wI = 0; wI < numberOfWeightsFromNextNeuron; wI ++ )
        {
            weights.add ( new Synapse ( ) );
            weights.get ( wI ).setWeight ( new Random ( ).nextDouble ( ) );
        }
    }
    
    
    //methods
    public ArrayList <Synapse> getWeights ( )
    {
        return weights;
    }
    public double getOutcome ( )
    {
        return outcome;
    }
    public double getGradient ( )
    {
        return gradient;
    }
    public double getActivation ( double value )
    {
        return Math.tanh ( value );
    }
    public double getPrimeActivation ( double value )
    {
        return 1 - Math.pow ( Math.tanh ( value ), 2 );
    }
    public double getDistributedWeightSigma ( Layer nextLayer )
    {
        double returnValue = 0;
        
        for ( int nLI = 0; nLI < nextLayer.size ( ) - 1; nLI ++ )
            returnValue += getWeights ( ).get ( nLI ).getWeight ( ) * nextLayer.get ( nLI ).getGradient ( );
            
        return returnValue;
    }
    
    public void setGradient ( double value )
    {
        gradient = value;
    }
    public void setOutcome ( double value )
    {
        outcome = value;
    }
    public void calculateHiddenGradient ( Layer nextLayer )
    {
        double delta = getDistributedWeightSigma ( nextLayer );
        
        setGradient ( getPrimeActivation ( outcome ) * delta );
    }
    public void calculateOutcomeGradient ( int target )
    {
        double delta = target - outcome;
        
        setGradient ( getPrimeActivation ( outcome ) * delta );
    }
   
    public void doForwardPropagation ( Layer priorLayer )
    {
        //forward prop
        double sigma = 0;
        
        for ( int pLI = 0; pLI < priorLayer.size ( ); pLI ++ )
            sigma += priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).getWeight ( ) * priorLayer.get ( pLI ).getOutcome ( );
            
        setOutcome ( getActivation ( sigma ) );
    }
    
    public void updateWeights ( Layer priorLayer )
    {
        for ( int pLI = 0; pLI < priorLayer.size ( ); pLI ++ )
        {
            double priorDeltaWeight = priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).getDeltaWeight ( );
            
            //( eta * grad * priorO ) + ( alpha * pDW )
            double newDeltaWeight = ( eta * gradient * priorLayer.get ( pLI ).getOutcome ( ) ) + ( alpha * priorDeltaWeight );
            
            
            priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).setDeltaWeight ( newDeltaWeight );
            priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).setWeight ( priorLayer.get ( pLI ).getWeights ( ).get ( neuronId ).getWeight ( ) + newDeltaWeight );
        }
    }
}
