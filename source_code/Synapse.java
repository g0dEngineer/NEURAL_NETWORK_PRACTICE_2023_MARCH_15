//Author: God Bennett , 2023, march 15
//Title: Neural network from scratch/memory

public class Synapse
{
    //features
    private double weight;
    private double deltaWeight;
    
    //methods
    public double getWeight ( )
    {
        return weight;
    }
    public double getDeltaWeight ( )
    {
        return deltaWeight;
    }
    public void setWeight ( double value )
    {
        weight = value;
    }
    public void setDeltaWeight ( double value )
    {
        deltaWeight = value;
    }
}
