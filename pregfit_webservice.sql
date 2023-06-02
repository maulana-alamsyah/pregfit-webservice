-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: May 07, 2023 at 11:45 AM
-- Server version: 10.4.11-MariaDB
-- PHP Version: 8.1.14

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `pregfit_webservice`
--

-- --------------------------------------------------------

--
-- Table structure for table `feedback`
--

CREATE TABLE `feedback` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `komentar` varchar(255) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `feedback`
--

INSERT INTO `feedback` (`id`, `user_id`, `komentar`, `created_at`) VALUES
(1, 1, 'Aku senang sekali badan terasa sehat dan gembira', '2023-04-02 07:17:22');

-- --------------------------------------------------------

--
-- Table structure for table `history`
--

CREATE TABLE `history` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `tanggal` date NOT NULL,
  `waktu` varchar(50) NOT NULL,
  `jenis_yoga` varchar(100) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `history`
--

INSERT INTO `history` (`id`, `user_id`, `tanggal`, `waktu`, `jenis_yoga`, `created_at`) VALUES
(1, 1, '2023-04-02', '10 Menit', 'Cat Cow Pose', '2023-04-02 02:11:36'),
(2, 1, '2023-04-02', '5 Menit', 'Bird Dog Pose', '2023-04-02 02:38:30');

-- --------------------------------------------------------

--
-- Table structure for table `user`
--

CREATE TABLE `user` (
  `id` int(11) NOT NULL,
  `no_hp` varchar(13) NOT NULL,
  `email` varchar(32) DEFAULT NULL,
  `nama` varchar(64) DEFAULT NULL,
  `usia_kandungan` varchar(15) DEFAULT NULL,
  `tanggal_lahir` date DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `user`
--

INSERT INTO `user` (`id`, `no_hp`, `email`, `nama`, `usia_kandungan`, `tanggal_lahir`, `created_at`, `updated_at`) VALUES
(1, '089636092727', 'maulanalamsyah@gmail.com', 'moms', 'Trisemester 2', '2003-02-01', '2023-03-29 16:26:42', '2023-04-15 13:33:29'),
(2, '089636092728', NULL, 'bunda', 'Trisemester 1', '2000-07-28', '2023-03-29 16:34:44', '2023-03-29 16:34:44'),
(3, '0897787654', NULL, NULL, NULL, NULL, '2023-03-30 04:00:09', '2023-03-30 04:00:09'),
(4, '0812345678', NULL, NULL, NULL, NULL, '2023-03-30 04:18:05', '2023-03-30 04:18:05'),
(5, '089636092929', NULL, NULL, NULL, NULL, '2023-04-15 12:53:07', '2023-04-15 12:57:10'),
(6, '089636092828', NULL, NULL, NULL, NULL, '2023-04-15 12:57:44', '2023-04-15 13:32:44'),
(7, '0821222233', NULL, NULL, NULL, NULL, '2023-04-29 13:41:04', '2023-04-29 13:41:04');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `feedback`
--
ALTER TABLE `feedback`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `history`
--
ALTER TABLE `history`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `user`
--
ALTER TABLE `user`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `no_hp` (`no_hp`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `feedback`
--
ALTER TABLE `feedback`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `history`
--
ALTER TABLE `history`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT for table `user`
--
ALTER TABLE `user`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
