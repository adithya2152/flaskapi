import React, { useState, useRef, useEffect } from 'react';
import { View, StyleSheet, ScrollView, Alert } from 'react-native';
import { Button, Card, Title, Text, Chip, ActivityIndicator } from 'react-native-paper';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { StackNavigationProp } from '@react-navigation/stack';
import { RouteProp } from '@react-navigation/native';
import { RootStackParamList } from '../../App';
import { ApiService } from '../services/api';
import AsyncStorage from '@react-native-async-storage/async-storage';

type MonitoringScreenNavigationProp = StackNavigationProp<RootStackParamList, 'Monitoring'>;
type MonitoringScreenRouteProp = RouteProp<RootStackParamList, 'Monitoring'>;

interface Props {
  navigation: MonitoringScreenNavigationProp;
  route: MonitoringScreenRouteProp;
}

export default function MonitoringScreen({ navigation, route }: Props) {
  const { username } = route.params;
  const [permission, requestPermission] = useCameraPermissions();
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [alerts, setAlerts] = useState<string[]>([]);
  const [lastAlert, setLastAlert] = useState<string>('');
  const cameraRef = useRef<CameraView>(null);
  const monitoringInterval = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    return () => {
      if (monitoringInterval.current) {
        clearInterval(monitoringInterval.current);
      }
    };
  }, []);

  if (!permission) {
    return <View />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Card style={styles.card}>
          <Card.Content style={styles.cardContent}>
            <Title>Camera Permission Required</Title>
            <Button mode="contained" onPress={requestPermission} style={styles.button}>
              Grant Permission
            </Button>
          </Card.Content>
        </Card>
      </View>
    );
  }

  const startMonitoring = async () => {
    setIsMonitoring(true);
    setAlerts([]);
    
    monitoringInterval.current = setInterval(async () => {
      if (cameraRef.current) {
        try {
          const photo = await cameraRef.current.takePictureAsync({
            base64: true,
            quality: 0.5,
          });

          if (photo?.base64) {
            const result = await ApiService.sendFrame(photo.base64, username);
            
            if (result.success && result.data?.alerts) {
              const newAlerts = result.data.alerts;
              if (newAlerts.length > 0) {
                setAlerts(prev => [...prev, ...newAlerts].slice(-20)); // Keep last 20 alerts
                setLastAlert(newAlerts[newAlerts.length - 1]);
              }
            }
          }
        } catch (error) {
          console.error('Monitoring error:', error);
        }
      }
    }, 2000); // Send frame every 2 seconds
  };

  const stopMonitoring = () => {
    setIsMonitoring(false);
    if (monitoringInterval.current) {
      clearInterval(monitoringInterval.current);
      monitoringInterval.current = null;
    }
  };

  const handleLogout = async () => {
    Alert.alert(
      'Logout',
      'Are you sure you want to logout?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Logout',
          onPress: async () => {
            stopMonitoring();
            await AsyncStorage.removeItem('username');
            navigation.navigate('Home');
          },
        },
      ]
    );
  };

  return (
    <View style={styles.container}>
      <Card style={styles.headerCard}>
        <Card.Content style={styles.headerContent}>
          <Title style={styles.welcomeText}>Welcome, {username}</Title>
          <Button mode="outlined" onPress={handleLogout} style={styles.logoutButton}>
            Logout
          </Button>
        </Card.Content>
      </Card>

      <Card style={styles.cameraCard}>
        <Card.Content style={styles.cameraContent}>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing={CameraType.front}
          />
          <View style={styles.controlsContainer}>
            {!isMonitoring ? (
              <Button
                mode="contained"
                onPress={startMonitoring}
                style={styles.startButton}
                contentStyle={styles.buttonContent}
              >
                Start Monitoring
              </Button>
            ) : (
              <Button
                mode="contained"
                onPress={stopMonitoring}
                style={[styles.startButton, styles.stopButton]}
                contentStyle={styles.buttonContent}
              >
                Stop Monitoring
              </Button>
            )}
          </View>
        </Card.Content>
      </Card>

      <Card style={styles.alertsCard}>
        <Card.Content>
          <View style={styles.alertsHeader}>
            <Title style={styles.alertsTitle}>Live Alerts</Title>
            {isMonitoring && <ActivityIndicator size="small" />}
          </View>
          
          {lastAlert ? (
            <Chip style={styles.lastAlertChip} textStyle={styles.lastAlertText}>
              {lastAlert}
            </Chip>
          ) : null}

          <ScrollView style={styles.alertsScroll} showsVerticalScrollIndicator={false}>
            {alerts.length === 0 ? (
              <Text style={styles.noAlertsText}>
                {isMonitoring ? 'Monitoring... No alerts yet' : 'Start monitoring to see alerts'}
              </Text>
            ) : (
              alerts.slice().reverse().map((alert, index) => (
                <Text key={index} style={styles.alertText}>
                  {alert}
                </Text>
              ))
            )}
          </ScrollView>
        </Card.Content>
      </Card>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f4f6fa',
    padding: 16,
  },
  card: {
    elevation: 4,
    borderRadius: 12,
    marginBottom: 16,
  },
  headerCard: {
    marginTop: 8,
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
  },
  welcomeText: {
    fontSize: 18,
    color: '#1976d2',
  },
  logoutButton: {
    borderRadius: 6,
  },
  cameraCard: {
    flex: 1,
  },
  cameraContent: {
    flex: 1,
    padding: 16,
  },
  camera: {
    flex: 1,
    borderRadius: 12,
    overflow: 'hidden',
  },
  controlsContainer: {
    marginTop: 16,
    alignItems: 'center',
  },
  startButton: {
    borderRadius: 8,
    minWidth: 180,
  },
  stopButton: {
    backgroundColor: '#d32f2f',
  },
  buttonContent: {
    paddingVertical: 8,
  },
  alertsCard: {
    maxHeight: 200,
  },
  alertsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  alertsTitle: {
    fontSize: 16,
    color: '#d32f2f',
  },
  lastAlertChip: {
    backgroundColor: '#ffebee',
    marginBottom: 12,
  },
  lastAlertText: {
    color: '#d32f2f',
    fontSize: 12,
  },
  alertsScroll: {
    maxHeight: 120,
  },
  alertText: {
    fontSize: 12,
    color: '#d32f2f',
    marginBottom: 4,
    fontFamily: 'monospace',
  },
  noAlertsText: {
    textAlign: 'center',
    color: '#757575',
    fontStyle: 'italic',
    marginTop: 20,
  },
  cardContent: {
    padding: 20,
    alignItems: 'center',
  },
  button: {
    borderRadius: 8,
    marginTop: 16,
  },
});</parameter>