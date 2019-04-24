/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import java.lang.reflect.InvocationTargetException;
import weka.core.Instance;
import weka.core.Instances;
import java.lang.reflect.Method;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.DenseInstance;
import weka.core.SparseInstance;

/**
 * Class to calculate average Hausdorff distance between bags.
 *
 * @author Danel
 */
public class AveHausdorffDistance {

  /**
   * Calculate the average Hausdorff distance between two bags.
   *
   * @param distance
   * @param A one bag
   * @param B the other bag
   * @return the distance value
   */
  public static double measure(Class<? extends InstanceLevelDistance> distance, Instance A, Instance B) {
    double m = 0;
    Instances bagA = A.relationalValue(1);
    Instances bagB = B.relationalValue(1);
    for (int i = 0; i < bagA.numInstances(); i++) {
      Instance x = bagA.instance(i);
      m += min(distance, x, B);
    }
    for (int i = 0; i < bagB.numInstances(); i++) {
      Instance x = bagB.instance(i);
      m += min(distance, x, A);
    }
    int N = bagA.numInstances() + bagB.numInstances();
    m /= N;
    return m;
  }
  
  public static double measure(Instance A, Instance B) {
    return measure(EuclideanDistance.class, A, B);
  }
  
  public static double euclideanMeasure(Instance A, Instance B) {
    double m = 0;
    Instances bagA = A.relationalValue(1);
    Instances bagB = B.relationalValue(1);
    for (int i = 0; i < bagA.numInstances(); i++) {
      Instance x = bagA.instance(i);
      m += euclideanMin(x, B);
    }
    for (int i = 0; i < bagB.numInstances(); i++) {
      Instance x = bagB.instance(i);
      m += euclideanMin(x, A);
    }
    int N = bagA.numInstances() + bagB.numInstances();
    m /= N;
    return m;
  }

  public static double denseEuclideanMeasure(Instance A, Instance B) {
    double m = 0;
    Instances bagA = A.relationalValue(1);
    Instances bagB = B.relationalValue(1);
    for (int i = 0; i < bagA.numInstances(); i++) {
      DenseInstance x = (DenseInstance)bagA.instance(i);
      m += denseEuclideanMin(x, B);
    }
    for (int i = 0; i < bagB.numInstances(); i++) {
      DenseInstance x = (DenseInstance)bagB.instance(i);
      m += denseEuclideanMin(x, A);
    }
    int N = bagA.numInstances() + bagB.numInstances();
    m /= N;
    return m;
  }

  public static double sparseEuclideanMeasure(Instance A, Instance B) {
    double m = 0;
    Instances bagA = A.relationalValue(1);
    Instances bagB = B.relationalValue(1);
    for (int i = 0; i < bagA.numInstances(); i++) {
      SparseInstance x = (SparseInstance)bagA.instance(i);
      m += sparseEuclideanMin(x, B);
    }
    for (int i = 0; i < bagB.numInstances(); i++) {
      SparseInstance x = (SparseInstance)bagB.instance(i);
      m += sparseEuclideanMin(x, A);
    }
    int N = bagA.numInstances() + bagB.numInstances();
    m /= N;
    return m;
  }

  public static double cosineMeasure(Instance A, Instance B) {
    double m = 0;
    Instances bagA = A.relationalValue(1);
    Instances bagB = B.relationalValue(1);
    for (int i = 0; i < bagA.numInstances(); i++) {
      Instance x = bagA.instance(i);
      m += cosineMin(x, B);
    }
    for (int i = 0; i < bagB.numInstances(); i++) {
      Instance x = bagB.instance(i);
      m += cosineMin(x, A);
    }
    int N = bagA.numInstances() + bagB.numInstances();
    m /= N;
    return m;
  }

  private static double euclideanMin(Instance a, Instance B) {
    double m = Double.MAX_VALUE;
    for (int i = 0; i < B.relationalValue(1).numInstances(); i++) {
      Instance x = B.relationalValue(1).instance(i);
      double s = EuclideanDistance.measure(a, x);
      if (s < m)
        m = s;
    }
    return m;
  }
  
  private static double denseEuclideanMin(DenseInstance a, Instance B) {
    double m = Double.MAX_VALUE;
    for (int i = 0; i < B.relationalValue(1).numInstances(); i++) {
      DenseInstance x = (DenseInstance)B.relationalValue(1).instance(i);
      double s = EuclideanDistance.denseMeasure(a, x);
      if (s < m)
        m = s;
    }
    return m;
  }
  
  private static double sparseEuclideanMin(SparseInstance a, Instance B) {
    double m = Double.MAX_VALUE;
    for (int i = 0; i < B.relationalValue(1).numInstances(); i++) {
      SparseInstance x = (SparseInstance)B.relationalValue(1).instance(i);
      double s = EuclideanDistance.sparseMeasure(a, x);
      if (s < m)
        m = s;
    }
    return m;
  }
  
  private static double cosineMin(Instance a, Instance B) {
    double m = Double.MAX_VALUE;
    for (int i = 0; i < B.relationalValue(1).numInstances(); i++) {
      Instance x = B.relationalValue(1).instance(i);
      double s = CosineDistance.measure(a, x);
      if (s < m)
        m = s;
    }
    return m;
  }
  
  private static double min(Class<? extends InstanceLevelDistance> distance, Instance a, Instance B) {
    double m = Double.MAX_VALUE;
    try {
      Method method = distance.getMethod("measure", new Class[] {Instance.class, Instance.class});
      for (int i = 0; i < B.relationalValue(1).numInstances(); i++) {
        Instance x = B.relationalValue(1).instance(i);
        double s = (Double)method.invoke(a, x);
        if (s < m)
          m = s;
      }
    } catch (NoSuchMethodException ex) {
      Logger.getLogger(AveHausdorffDistance.class.getName()).log(Level.SEVERE, null, ex);
    } catch (SecurityException ex) {
      Logger.getLogger(AveHausdorffDistance.class.getName()).log(Level.SEVERE, null, ex);
    } catch (IllegalAccessException ex) {
      Logger.getLogger(AveHausdorffDistance.class.getName()).log(Level.SEVERE, null, ex);
    } catch (IllegalArgumentException ex) {
      Logger.getLogger(AveHausdorffDistance.class.getName()).log(Level.SEVERE, null, ex);
    } catch (InvocationTargetException ex) {
      Logger.getLogger(AveHausdorffDistance.class.getName()).log(Level.SEVERE, null, ex);
    }
    return m;
  }

}
