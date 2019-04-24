/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import weka.core.Instance;

/**
 * Abstract class to be extended by all classes that calculate measure between instances.
 *
 * @author Danel
 */
public abstract class InstanceLevelDistance {

  /**
   * Calculate a distance measure between two instance.
   * 
   * @param a one instance
   * @param b the other instance
   * @return the distance value
   */
  public static double measure(Instance a, Instance b) {
    return 0;
  }

}
