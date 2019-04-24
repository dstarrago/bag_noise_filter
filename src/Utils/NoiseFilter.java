/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Utils;

/**
 * Interface to a class which can identify noisy objects.
 * 
 * @author Danel
 */
public interface NoiseFilter {
  
  /**
   * Gets the decision of the filter for a specific object.
   * 
   * @param index the index of the object. This index is a zero-based 
   * counting of objects of the original data set (i.e., before filtering).
   * 
   * @return true if the object is identified as noisy and false otherwise.
   */
  public boolean decision(int index);
  
  /**
   * Gets the number of objects to which the filter is applied.
   * 
   * @return the number of objects.
   */
  public int numObjects();
  
  /**
   * Gets the number of objects that are predicted as noise by the filter.
   * 
   * @return the number of filtered objects.
   */
  public int numFiltered();

}
