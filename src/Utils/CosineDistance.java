/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.SparseInstance;

/**
 * Class to calculate cosine distance between instances.
 *
 * @author Danel
 */
public class CosineDistance extends InstanceLevelDistance {

  /**
   * Calculate the cosine distance between two instance.
   * 
   * @param a one instance
   * @param b the other instance
   * @return the distance value
   */
  public static double measure(Instance a, Instance b) {
    if (a instanceof SparseInstance)        // Asumo que b tambien es SparseInstance
      return 1 - sparseCosSimilarity((SparseInstance)a, (SparseInstance)b);
    else
      return 1 - denseCosSimilarity((DenseInstance)a, (DenseInstance)b);
  }

  public static double sparseCosSimilarity(SparseInstance a, SparseInstance b) {
    double ab = 0, a2 = 0, b2 = 0;
    for (int i = 0; i < a.numValues(); i++) {
      a2 += a.valueSparse(i) * a.valueSparse(i);
    }
    for (int i = 0, bi; i < b.numValues(); i++) {
      bi = b.index(i);
      b2 += b.valueSparse(i) * b.valueSparse(i);
      double av = a.value(bi);
      if (av != 0)
        ab += av * b.valueSparse(i);
    }
    if (a2 == 0 && b2 == 0) return 1;                   // both instances are null
    if (a2 == 0 || b2 == 0) return 0;                   // one instance is null and the other does not
    double s = (ab / (Math.sqrt(a2) * Math.sqrt(b2)));  // neither of the two instance is null
    return (s > 1)? 1 : s;
  }

  public static double denseCosSimilarity(DenseInstance a, DenseInstance b) {
    double ab = 0, a2 = 0, b2 = 0;
    for (int i = 0; i < a.numAttributes(); i++) {
      ab += a.value(i) * b.value(i);
      a2 += a.value(i) * a.value(i);
      b2 += b.value(i) * b.value(i);
    }
    if (a2 == 0 && b2 == 0) return 1;                   // both instances are null
    if (a2 == 0 || b2 == 0) return 0;                   // one instance is null and the other does not
    double s = (ab / (Math.sqrt(a2) * Math.sqrt(b2)));  // neither of the two instance is null
    return (s > 1)? 1 : s;                              // Arithmetic precision correction
  }

}
