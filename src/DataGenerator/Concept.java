/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package DataGenerator;

import java.util.Random;

/**
 * Class representing a Gaussian concept in the instance space.
 * All covariances of concept distribution are 1.
 * 
 * @author Danel
 */
public class Concept {
  
  /**
   * Number of dimensions of the instance space.
   */
  private final int numDimensions;
  
  /**
   * Coordinates of the concept center.
   */
  private final double[] coords;
  
  /**
   * Random numbers generator.
   */
  private final Random random;
  
  public Concept(double... coords) {
    numDimensions = coords.length;
    this.coords = coords;
    random = new Random();
  }
  
  public double[] sample() {
    double[] vals = new double[numDimensions];
    for (int i = 0; i < numDimensions; i++) {
      vals[i] = coords[i] + random.nextGaussian();
    }
    return vals;
  }
  
  public int numDimensions() {
    return numDimensions;
  }
  
}
