/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Utils;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.SparseInstance;

/**
 *
 * @author Danel
 */
public class EuclideanDistance extends InstanceLevelDistance {
  
  public static double measure(Instance a, Instance b) {
    if (a instanceof SparseInstance)        // Asumo que b tambien es SparseInstance
      return sparseMeasure((SparseInstance)a, (SparseInstance)b);
    else
      return denseMeasure((DenseInstance)a, (DenseInstance)b);
  }
  
  public static double sparseMeasure(SparseInstance a, SparseInstance b) {
    double s = 0;
    double dif, a2 = 0, b2 = 0;
    for (int i = 0; i < a.numValues(); i++) {
      a2 += a.valueSparse(i) * a.valueSparse(i);
      double bv = b.value(a.index(i));
      dif = bv - a.valueSparse(i);
      s += dif * dif;
    }
    for (int i = 0; i < b.numValues(); i++) {
      b2 += b.valueSparse(i) * b.valueSparse(i);
      double av = a.value(b.index(i));
      if (av == 0) {
        dif = b.valueSparse(i);
        s += dif * dif;
      }
    }
    if (s == 0) return 0;                   
    double D = Math.sqrt(a2) + Math.sqrt(b2);
    return Math.sqrt(s) / D;                
  }

  public static double denseMeasure(DenseInstance a, DenseInstance b) {
    double s = 0;
    double dif, a2 = 0, b2 = 0;
    for (int i = 1; i < a.numAttributes() - 1; i++) {
      dif = a.value(i) - b.value(i);
      s += dif * dif;
      a2 += a.value(i) * a.value(i);
      b2 += b.value(i) * b.value(i);
    }
    if (s == 0) return 0;                   
    double D = Math.sqrt(a2) + Math.sqrt(b2);
    return Math.sqrt(s) / D;                
  }
  
}
